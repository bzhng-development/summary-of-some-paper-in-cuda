# YaRN: Efficient Context Window Extension of Large Language Models

**ArXiv:** [2309.00071](https://arxiv.org/abs/2309.00071)

## 🎯 Pitch

YaRN introduces a breakthrough method for vastly extending the usable context window of transformer models using Rotary Position Embeddings, such as LLaMA and Mistral. By combining a targeted rescaling of positional frequencies with a simple attention scaling trick, YaRN enables models to handle ultra-long sequences (up to 128k tokens) with 10× less training data and 2.5× fewer steps than previous state-of-the-art, all while retaining strong short-context performance. This innovation unlocks practical scalability for large language models in real-world scenarios that demand processing long documents or dialogues, without the prohibitive costs of model retraining.

---

## 1. Executive Summary

This paper introduces **YaRN (Yet another RoPE extensioN method)**, a compute-efficient method for extending the context window of transformer-based language models that use Rotary Position Embeddings (RoPE), requiring 10× fewer tokens and 2.5× fewer training steps than previous methods. The approach combines two main mechanisms — "NTK-by-parts" interpolation (which selectively interpolates RoPE frequencies based on their wavelength relative to the pretrained context length, rather than stretching all dimensions equally) and a temperature-based attention scaling (which modulates the softmax logits to maintain stable perplexity across the extended window) — and is evaluated on LLaMA and Llama 2 models (7B and 13B parameters) using long-sequence language modeling on Proof-pile and GovReport datasets, passkey retrieval, and standardized benchmarks. YaRN achieves 4× efficiency over Position Interpolation (matching PI's fine-tuned performance with 400 training steps instead of 1,000) and successfully extends Llama 2 models to 128k context length — a 32× increase over the original 4k pretrained window — while preserving the original short-context benchmark performance with minimal degradation (averaging 0.49% score drop between s = 16 and s = 32 extensions), establishing that effective context window extension is possible through targeted frequency-dependent interpolation only when specific RoPE dimensions encoding absolute positional information are interpolation-targeted while those encoding relative positional information are preserved intact.

## 2. Context and Motivation

### The Core Problem: Transformer Models Cannot Generalize Past Their Training Context Length

The fundamental problem this paper addresses is deceptively simple: **transformer-based language models trained with Rotary Position Embeddings (RoPE) fail catastrophically when presented with sequences longer than their pretraining context window**. A model trained on 2,048 or 4,096 tokens cannot process a 32,768-token document without its outputs degrading to nonsense — perplexity explodes, attention patterns break down, and the model effectively loses the ability to do anything useful with the extra tokens. This is not a gradual degradation; it is an abrupt cliff beyond the trained length. The paper's central goal is to find a method that extends this context window — ideally by an order of magnitude or more — using minimal additional training, without modifying the transformer architecture itself, and without sacrificing performance on short sequences.

### Why This Problem Matters

Context window length is not a minor implementation detail — it fundamentally constrains what language models can do. Several downstream capabilities are directly gated by how much text the model can attend to at once:

- **In-context learning (ICL)** : Few-shot prompting, where the model learns from examples provided in the prompt, requires fitting all those examples plus the query into the context window. More context means more examples, which improves ICL performance.
- **Long-document tasks** : Summarization of lengthy reports, question-answering over book-length texts, and analysis of multi-document collections all require the model to hold large amounts of text in working memory simultaneously. A 4k-token window (roughly 3,000 words) is insufficient for most real-world documents.
- **Multi-turn dialogue** : Conversational agents accumulate conversation history over time. A short context window forces either truncation of earlier turns (losing important context) or expensive retrieval-based workarounds.
- **Code generation and analysis** : Software projects routinely exceed tens of thousands of tokens. Understanding a codebase requires attending across files and functions that span far more than 4k tokens.
- **Retrieval-augmented generation** : When models are given retrieved documents as context, the amount of retrievable information is bounded by the context window. Larger windows enable more comprehensive retrieval.

The problem is also **theoretically significant** because it touches on a fundamental question about transformer architectures: can the positional encoding scheme that works during training generalize to unseen lengths, or is length generalization inherently limited by how position information is represented? Answering this has implications beyond just RoPE-based models — it speaks to whether transformers can ever be "truly" length-generalizing or whether length extension will always require some form of architectural intervention or retraining.

### Prior Approaches and Where They Fall Short

Before YaRN, several approaches to context window extension existed, each with significant limitations:

#### Direct Extrapolation (No Interpolation, No Fine-tuning)

The simplest approach is to do nothing — just feed the model longer sequences and hope it generalizes. This fails immediately. As the paper notes in Section A.1, "a direct extrapolation does not perform well on sequences... with L larger than the pre-trained limit." The reason is that RoPE's rotary frequencies are fixed during training; positions beyond the training maximum produce rotation angles the model has never seen, causing the attention mechanism to produce essentially random patterns. Perplexity explodes to >100 within a few thousand tokens past the limit.

#### Position Interpolation (PI) — Uniform Stretching

Proposed by Chen et al. (2023) and concurrently by kaiokendev (2023), PI was the first method to demonstrate that context extension was feasible with modest fine-tuning. The idea is simple: multiply all position indices by a scale factor $s = L'/L$ (where $L$ is the pretrained length and $L'$ is the target length) so that the maximum position at length $L'$ maps back to $L$ — exactly the position range seen during pretraining. Under the notation the paper establishes in Equation 7, PI sets $g(m) = s \cdot m$ and leaves the frequency basis unchanged ($h(\theta) = \theta$).

This works, but with sharp limitations:

- **It destroys high-frequency information.** By stretching *every* RoPE dimension equally by $s$, PI compresses all rotary frequencies. The high-frequency dimensions that encode fine-grained relative position differences between nearby tokens get stretched out, losing their ability to distinguish adjacent tokens from tokens that are $s$ positions apart. This is a fundamental signal-processing problem: you cannot stretch a waveform without losing its high-frequency components.
- **It only scales to about $s = 8$.** The paper reports that "previous fine-tunes using PI were only able to achieve a scaling factor of roughly $s = 8$ before the LLM's outputs starts to degrade, even after fine-tuning." This means a 4k-context model could only reach ~32k — useful, but far from the 128k+ that practical applications demand.
- **It requires substantial fine-tuning.** Chen et al. (2023) used approximately 1,000 training steps and billions of tokens to adapt models to the interpolated positions. The paper's Table 4 shows that Chen et al.'s PI extension to 8× context required 640 A100 GPU-hours for a 7B model — a non-trivial computational cost.

The fundamental flaw in PI is its **uniform treatment of all RoPE dimensions**. As the paper later explains (Section 3.2), different RoPE dimensions serve fundamentally different roles: some encode primarily relative position (short wavelengths), others encode primarily absolute position (long wavelengths). Stretching them all equally damages the relative-position dimensions that the model relies on for local attention patterns.

#### "NTK-aware" Interpolation — Base Frequency Change

An improvement proposed by bloc97 (2023a), the "NTK-aware" method draws on Neural Tangent Kernel theory (Tancik et al., 2020) to argue that deep networks struggle to learn high-frequency information when the input lacks high-frequency components. Rather than scaling all frequencies uniformly, it changes the base of the RoPE exponential from $b = 10000$ to $b' = b \cdot s^{|D|/(|D|-2)}$, where $|D|$ is the hidden dimension size. The effect is to **spread the interpolation pressure unevenly across dimensions**: low frequencies (which correspond to long-range positions) get scaled more, while high frequencies (short-range) are scaled less, preserving more of the fine-grained positional information that PI destroys.

This was a significant empirical improvement over PI for non-fine-tuned models and was adopted by Code Llama (Rozière et al., 2023) under the name "adjusted base frequency." However, it suffers from two critical weaknesses:

- **The optimal base is unknown a priori.** The paper notes that "it is very difficult to determine what optimal base should be used for an intended context extension by $s$ times. The best base to use for 'NTK-aware' interpolation usually has to be found empirically, which significantly increases the difficulty and cost of obtaining a successful fine-tuned model." The formula $b' = b \cdot s^{|D|/(|D|-2)}$ provides a starting point but is not reliably optimal — practitioners must sweep values, adding costly hyperparameter tuning to an already expensive process.
- **It performs worse than PI when fine-tuned.** The paper reports that "fine-tuning with 'NTK-aware' interpolation yields inferior results to PI." This is a crucial weakness: the method that works better without fine-tuning works worse with it. Since fine-tuning is the path to the largest extension factors, this limits the method's practical utility for aggressive scaling.
- **Some dimensions are extrapolated, not interpolated.** Because the base change is a continuous transformation, some dimensions end up with frequencies that effectively correspond to positions beyond the pretrained range — they are "extrapolated" rather than interpolated. This means "the theoretical scale factor $s$ does not accurately describe the true context extension scale" — the actual effective extension is different from the nominal $s$, making the method unpredictable.

#### "NTK-by-parts" Interpolation — Wavelength-Based Selective Interpolation

Proposed by bloc97 (2023b) as a further refinement, this method makes the insight behind "NTK-aware" interpolation explicit and targeted. Rather than using a continuous base change that approximates the desired behavior, it directly classifies each RoPE dimension based on its **wavelength** $\lambda_d = 2\pi / \theta_d$ relative to the pretrained context length $L$:

- Dimensions where the wavelength is **much shorter than** $L$ (high frequency, many full rotations during training) encode primarily **relative positional information** — they help the model distinguish token 5 from token 6, not token 5 from token 5000. These should **not be interpolated at all**, because stretching them would destroy the fine-grained relative position signal.
- Dimensions where the wavelength is **equal to or longer than** $L$ (low frequency, less than one full rotation during training) encode primarily **absolute positional information** — they help the model know *where* in the sequence a token is. These **must be interpolated** to avoid out-of-distribution positions, exactly as PI does.
- Dimensions with intermediate wavelengths should receive a **partial interpolation**, smoothly transitioning between the two regimes via a ramp function.

This is the first method that explicitly accounts for the **dual nature of RoPE** — it simultaneously encodes both relative and absolute position, and different dimensions serve different roles. The paper formalizes this using a ratio $r(d) = L / \lambda_d$ that measures how many full rotations dimension $d$ makes during a sequence of length $L$, and introduces a ramp function $\gamma(r)$ (Equation 11) with two threshold parameters $\alpha$ and $\beta$ that define the interpolation boundaries.

"NTK-by-parts" outperforms both PI and "NTK-aware" interpolation, but it was developed independently and lacks the attention-scaling component that YaRN adds. It is a direct precursor to YaRN — YaRN is essentially "NTK-by-parts" plus the temperature-based attention scaling described in Section 3.3.

#### Other Approaches: ReRoPE and LM-Infinite

The paper briefly acknowledges two concurrent approaches that are **not pure embedding interpolation methods**:

- **ReRoPE** (Su, 2023) modifies the attention mechanism itself and claims "infinite" context length without fine-tuning. However, it "is currently not compatible with Flash Attention 2 and requires two attention passes during inference," making it impractical for most production deployments where Flash Attention's speed and memory savings are critical.
- **LM-Infinite** (Han et al., 2023) proposes similar ideas to YaRN but focuses on non-fine-tuned models and also modifies the attention mechanism, making it similarly incompatible with Flash Attention 2.

Both approaches highlight a key advantage of YaRN: by working purely at the level of rotary position embeddings (not the attention computation itself), YaRN remains **fully compatible with Flash Attention 2 and other attention optimizations, and incurs zero additional computational cost during inference**.

#### The Gap: No Method Efficiently Reaches Very Long Contexts While Preserving Short-Context Performance

By the time of YaRN's development, existing methods could extend context windows, but none simultaneously achieved:

- **Large extension factors** ($s = 32$ or more, going from 4k to 128k)
- **Minimal training cost** (hundreds of steps, not thousands, and 10× fewer tokens than PI)
- **Preserved short-context performance** (no significant regression on standard benchmarks)
- **Architectural compatibility** (works with Flash Attention 2, requires no attention mechanism changes)
- **Predictable, principled design** (no opaque hyperparameter sweeps to find the right base frequency)

The paper positions YaRN as filling exactly this gap: it is the convergence point of the evolutionary line from PI → "NTK-aware" → "NTK-by-parts", with the addition of temperature-based attention scaling that stabilizes perplexity across the extended window.

### How This Paper Positions Itself

The paper frames its contribution not as a completely novel idea but as the **synthesis and refinement** of a lineage of methods, culminating in a practical recipe that works at scales ($s = 32$, 128k context) that prior methods could not reach. Figure 1 explicitly traces the genealogy: Position Interpolation → "NTK-aware" → "NTK-by-parts" → YaRN, with Dynamic Scaling as a complementary inference-time technique that branches off from "NTK-aware."

This positioning is honest and useful — it makes clear that YaRN is not a radical departure but rather the result of understanding *why* each prior method succeeded or failed and systematically addressing those failure modes. The paper's contributions are:

1. **A principled framework for understanding RoPE dimensions** (Section 3.2): the insight that wavelength relative to context length determines whether a dimension encodes relative or absolute position, and that these two types must be treated differently during interpolation.
2. **The temperature-based attention scaling** (Section 3.3): the empirical discovery that modulating the softmax temperature by a factor $t$ that scales with the extension factor $s$ (specifically, $\sqrt{1/t} = 0.1 \ln(s) + 1$) uniformly improves perplexity across all positions in the extended window, and that this can be implemented as a zero-cost scaling of the rotary embeddings themselves rather than modifying the attention computation.
3. **The demonstration that these components work in concert** to achieve extension factors ($s = 32$, or 128k context from 4k pretraining) that were previously unattainable, while using 10× fewer training tokens and 2.5× fewer training steps than PI.
4. **The Dynamic Scaling inference-time technique** (Section 3.4) that allows models to gracefully handle variable-length sequences up to and beyond the fine-tuned extension limit by dynamically adjusting the scale factor per forward pass.

The paper also implicitly establishes a **philosophy about context extension**: that it should be a "drop-in replacement" requiring no architectural changes, no attention mechanism modifications, and no additional inference cost. This contrasts with approaches like ReRoPE and LM-Infinite that trade off compatibility for their claimed benefits. YaRN's commitment to being a pure embedding-level method — modifying only how the rotary position embeddings are computed, not how they are used — is a deliberate design choice that prioritizes practical deployability.

Finally, the paper positions its efficiency claims quantitatively against known baselines. Table 4 provides specific GPU-hour comparisons: YaRN achieves 64k context on Llama 2 7B in 256 A100-hours and 128k in 256+128 hours, compared to 640 hours for PI at only 8× extension (16k context), and 6,400–64,000 hours for other methods. The 10× token efficiency and 2.5× step efficiency claims are benchmarked against PI's known training requirements (Section 4.1: 400 steps for YaRN vs. 1,000 steps for PI to achieve comparable perplexity, as shown in Table 6). These are not abstract claims — they are directly measured and compared.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops a method — YaRN — that modifies how a transformer language model computes position information so that the model can process far longer sequences than it was originally trained on, without changing the model architecture and with minimal additional training. The problem: RoPE-based transformers catastrophically fail on sequences longer than their pretraining context window because the rotary position embeddings produce angles the model has never seen. The solution's shape: two complementary modifications to RoPE — (1) a **frequency-dependent interpolation** that treats each RoPE dimension differently based on whether it encodes relative or absolute position, and (2) a **temperature-based attention scaling** that stabilizes the softmax distribution across the extended window — combined with an optional inference-time technique (Dynamic Scaling) that adjusts the interpolation factor per forward pass to gracefully handle variable-length sequences.

### 3.2 Big-Picture Architecture (Diagram in Words)

The YaRN system modifies the position encoding pipeline at a single, narrow interface: the computation of rotary position embeddings before they are applied to query and key vectors in attention. The architecture has four conceptual components:

1. **Base LLM with RoPE** (e.g., LLaMA or Llama 2) — the pretrained transformer model whose attention layers use rotary position embeddings to encode token positions. This model is not architecturally modified; only the position embedding computation changes.
2. **"NTK-by-parts" Interpolation Module** — takes the original RoPE frequency basis and the target extension scale $s = L'/L$, classifies each hidden dimension into interpolation regimes based on its wavelength relative to the pretrained context length $L$, and produces a modified frequency for each dimension using a ramp-weighted combination of interpolated and non-interpolated values.
3. **Attention Temperature Scaling** — modifies the effective softmax temperature by a factor $t$ that depends on the extension scale $s$, implemented by scaling the rotary position embeddings themselves by $\sqrt{1/t}$ (exploiting the mathematical equivalence between pre-softmax temperature scaling and embedding magnitude scaling in RoPE).
4. **Dynamic Scaling (inference-time only)** — an optional wrapper that recomputes the scale factor $s = \max(1, l'/L)$ on each forward pass based on the current sequence length $l'$, allowing the model to gracefully handle sequences from length 1 to beyond the trained extension limit without performance cliffs.

Information flows as follows: at each forward pass, the current sequence length determines $s$ → the "NTK-by-parts" module uses $s$ and the per-dimension wavelength ratio $r(d)$ to compute modified frequencies $\tilde{\theta}_d$ → the attention temperature scaling further modifies these by multiplying by $\sqrt{1/t}$ → the resulting rotary embeddings are applied to query and key vectors as in standard RoPE → attention proceeds unchanged. During training, $s$ is fixed to the target extension factor; during inference with Dynamic Scaling, $s$ varies per forward pass.

### 3.3 Roadmap for the Deep Dive

- **First**, the RoPE formalism (Equations 1–7): what RoPE actually computes, the complex-number representation, and the mathematical interface that all interpolation methods modify. This establishes the shared notation that the "NTK-by-parts" and YaRN methods plug into.
- **Second**, the wavelength concept (Equation 8) and the ratio $r(d) = L/\lambda_d$ (Equation 10): why different RoPE dimensions encode fundamentally different types of positional information (relative vs. absolute), and how the wavelength determines which type a dimension provides.
- **Third**, the "NTK-by-parts" interpolation itself (Equations 11–13): the ramp function $\gamma(r)$, the per-dimension frequency modification $h(\theta_d)$, and the choice of thresholds $\alpha$ and $\beta$. This is the core interpolation engine that YaRN inherits.
- **Fourth**, the YaRN attention scaling (Equations 14–15): the temperature factor $t$, how it is implemented by scaling rotary embeddings, the empirical formula $\sqrt{1/t} = 0.1 \ln(s) + 1$, and why this stabilizes perplexity uniformly across the extended context.
- **Fifth**, Dynamic Scaling (Section 3.4): how varying $s$ per forward pass prevents performance cliffs, the interaction with KV-caching, and the correct implementation order.
- **Sixth**, the training procedure (Section 4.1): dataset, hyperparameters, the two-phase training for $s = 32$ (starting from the $s = 16$ checkpoint), and why this transfer-learning approach works.

This order traces the intellectual genealogy from Figure 1: PI → NTK-aware → NTK-by-parts → YaRN, with each step addressing a failure mode of its predecessor.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **methods paper** whose core idea is that context window extension of RoPE-based models requires treating different RoPE dimensions differently — interpolating those that encode absolute position while preserving those that encode relative position — and that an additional attention-temperature modification stabilizes perplexity across the extended window.

---

#### RoPE Formalism: The Mathematical Interface

The paper operates on RoPE exactly as defined in Su et al. (2022). Understanding the interface is essential because every interpolation method — PI, NTK-aware, NTK-by-parts, and YaRN — modifies RoPE at precisely the same point: the functions $g(m)$ and $h(\theta)$ in Equation 7.

**Attention computation with RoPE.** Given a sequence of hidden states $\mathbf{x}_1, \ldots, \mathbf{x}_L \in \mathbb{R}^{|D|}$, the attention layer first projects each hidden state into query and key vectors:

$$\mathbf{q}_m = f_q(\mathbf{x}_m, m) \in \mathbb{R}^{|D|}, \quad \mathbf{k}_n = f_k(\mathbf{x}_n, n) \in \mathbb{R}^{|D|}$$

where $m$ and $n$ are the position indices of the query and key tokens respectively, and $f_q, f_k$ are the position-encoding functions. Attention weights are then computed as:

$$\text{softmax}\left(\frac{\mathbf{q}_m^T \mathbf{k}_n}{\sqrt{|D|}}\right)$$

where $\mathbf{q}_m^T \mathbf{k}_n$ is the Euclidean inner product (dot product) between the query and key vectors.

**RoPE in complex coordinates.** RoPE assumes $|D|$ is even and identifies $\mathbb{R}^{|D|}$ with $\mathbb{C}^{|D|/2}$ via an isomorphism that pairs adjacent dimensions into complex numbers:

$$(x_1, x_2, \ldots, x_{|D|-1}, x_{|D|}) \mapsto (x_1 + i x_2, \ldots, x_{|D|-1} + i x_{|D|})$$

Under this identification, the dot product $\mathbf{q}^T \mathbf{k}$ becomes the real part of the Hermitian inner product $\text{Re}(\mathbf{q}^* \mathbf{k})$ in complex space. The key operation in RoPE is multiplying each complex coordinate by a rotation factor $e^{i m \theta_d}$, where $\theta_d = b^{-2d/|D|}$ with $b = 10000$ and $d$ indexing over the $|D|/2$ complex dimensions:

$$f_W(\mathbf{x}_m, m, \boldsymbol{\theta}) = e^{i m \boldsymbol{\theta}} W \mathbf{x}_m$$

where $W$ is a learned linear projection ($W_q$ for queries, $W_k$ for keys) and $e^{i m \boldsymbol{\theta}}$ acts coordinate-wise: the $d$-th complex coordinate of $W\mathbf{x}_m$ is multiplied by $e^{i m \theta_d}$.

**What this achieves.** Because $e^{i m \theta_d} \cdot \overline{e^{i n \theta_d}} = e^{i (m-n) \theta_d}$, the inner product between a query at position $m$ and a key at position $n$ depends only on their relative distance $m - n$, not on their absolute positions — this is what makes RoPE a **relative** position embedding. The set of frequencies $\{\theta_d\}$ controls which relative distances get emphasized by attention: high $\theta_d$ (small $d$, high frequency) oscillate rapidly with $m-n$, encoding fine-grained local distinctions; low $\theta_d$ (large $d$, low frequency) oscillate slowly, encoding coarse long-range distinctions.

**The universal modification interface.** All interpolation methods the paper discusses modify RoPE by replacing $f_W$ with:

$$f'_W(\mathbf{x}_m, m, \boldsymbol{\theta}) = f_W(\mathbf{x}_m, g(m), h(\boldsymbol{\theta}))$$

where $g: \mathbb{R} \to \mathbb{R}$ modifies the position index and $h$ acts coordinate-wise on the frequency vector $\boldsymbol{\theta}$, producing $h(\theta_d)$ for each dimension $d$. Different methods specify different $g$ and $h$:

- **Position Interpolation (PI):** $g(m) = m/s$, $h(\theta_d) = \theta_d$ — downscales every position by $s$, effectively compressing all frequencies uniformly.
- **NTK-aware:** $g(m) = m$, $h(\theta_d) = (b')^{-2d/|D|}$ with $b' = b \cdot s^{|D|/(|D|-2)}$ — changes the frequency base, which stretches low frequencies more than high frequencies.
- **NTK-by-parts (and YaRN's interpolation component):** $g(m) = m$, $h(\theta_d)$ is a per-dimension blend between $\theta_d/s$ (interpolated) and $\theta_d$ (preserved), controlled by the wavelength-based ramp function.
- **YaRN (full):** combines NTK-by-parts' $h(\theta_d)$ with an additional scaling of the rotary embeddings by $\sqrt{1/t}$ (which is mathematically equivalent to modifying the softmax temperature).

The $g(m) = m$ choice in NTK-aware, NTK-by-parts, and YaRN means these methods do not modify position indices — they only modify the frequencies, keeping the "position" axis unchanged. This is a deliberate design choice that preserves the model's ability to handle position-index-sensitive operations.

---

#### Wavelength and the Relative-vs-Absolute Distinction

The key insight driving the NTK-by-parts method is that RoPE dimensions with different wavelengths serve fundamentally different purposes. The paper formalizes this using the **wavelength** of each RoPE dimension:

$$\lambda_d = \frac{2\pi}{\theta_d} = 2\pi b^{2d/|D|}$$

where $b = 10000$, $|D|$ is the hidden dimension size, and $d \in \{0, 1, \ldots, |D|/2 - 1\}$ indexes the complex dimensions.

**What the wavelength means operationally.** $\lambda_d$ is the number of tokens required for the rotary embedding at dimension $d$ to complete one full rotation ($2\pi$ radians). For small $d$ (early dimensions), $\lambda_d$ is small — these dimensions oscillate rapidly with position, completing many full rotations within the pretrained context length $L$. For large $d$ (later dimensions), $\lambda_d$ is large — these dimensions may complete less than one full rotation over the entire pretraining context.

**The crucial observation from Section 3.2.** Given a pretrained context length $L$, there are dimensions $d$ where $\lambda_d > L$ — the wavelength is longer than the maximum sequence length seen during training. In such dimensions, the rotary embedding never completes a full rotation during pretraining. This means:

> "this suggests that some dimensions' rotary embeddings might not be distributed evenly in the rotational domain (i.e. does not perform a full rotation for the entire training context size). In such cases, we presume having unique position pairs implies that the absolute positional information remains intact in those dimensions."

In plain language: if a sine wave doesn't complete even one cycle over the full training length, then every position in the training data maps to a different point on that wave. The model can use this to determine absolute position — token 500 is at a different phase than token 2000, and the model has seen both during training, so it learns that this phase difference corresponds to that absolute distance. These dimensions effectively behave like **absolute position encodings**.

Conversely, when $\lambda_d$ is much smaller than $L$, the wave completes many cycles during training. Each absolute position maps to a phase that has been seen many times at other absolute positions (since the wave repeats). The model cannot use this dimension for absolute position — but it can use the *difference* between two positions' phases to determine their *relative* distance, since RoPE's dot product depends on $e^{i(m-n)\theta_d}$, which encodes relative position directly. These dimensions behave like **relative position encodings**.

**The ratio $r(d)$ quantifies this.** The paper introduces:

$$r(d) = \frac{L}{\lambda_d} = \frac{L}{2\pi} b^{-2d/|D|}$$

This is the number of full rotations (in units of $2\pi$) that dimension $d$ completes over the pretrained context length $L$. A small $r(d)$ (much less than 1) means few rotations — this dimension encodes absolute position. A large $r(d)$ (much greater than 1) means many rotations — this dimension encodes relative position.

**Why this distinction matters for interpolation.** When extending the context window from $L$ to $L' = s \cdot L$:

- Dimensions with large $r(d)$ (relative-position encoders) are **already robust** to longer sequences because they encode relative distances — the model uses $e^{i(m-n)\theta_d}$, and as long as the *relative* distances $(m-n)$ it encounters during extended-context inference are within the range it saw during training, these dimensions generalize. Interpolating them (as PI does) would *compress* their frequencies, destroying the fine-grained relative-position signal that is their primary function. **These dimensions should not be interpolated.**
- Dimensions with small $r(d)$ (absolute-position encoders) are **fragile** to longer sequences because the absolute positions $m > L$ map to phases the model has never seen. Without interpolation, the model encounters out-of-distribution phase angles and fails. **These dimensions must be interpolated** (mapping the new extended positions back into the training range, exactly as PI does).
- Dimensions with intermediate $r(d)$ should receive **partial interpolation**, smoothly transitioning between the two regimes.

This wavelength-based analysis is what distinguishes NTK-by-parts (and YaRN) from all prior methods. PI treats all dimensions identically (wrong for relative-position dimensions). NTK-aware approximates this behavior through a continuous base change (better, but imprecise and requires empirical tuning). NTK-by-parts makes the distinction explicit and deterministic.

---

#### The "NTK-by-parts" Interpolation

The NTK-by-parts method implements the wavelength-based selective interpolation using a **ramp function** that smoothly transitions between three regimes.

**Thresholds $\alpha$ and $\beta$.** The paper introduces two hyperparameters that partition the ratio $r(d)$ space:

- $\alpha$: dimensions with $r(d) < \alpha$ are fully interpolated (like PI) — these are the absolute-position dimensions.
- $\beta$: dimensions with $r(d) > \beta$ are not interpolated at all — these are the relative-position dimensions.
- $\alpha \leq r(d) \leq \beta$: dimensions receive a linear blend between the two regimes.

For LLaMA-family models, the paper reports experimentally determined values: $\alpha = 1$ and $\beta = 32$.

**The ramp function $\gamma(r)$.** This function defines the interpolation fraction for each dimension:

$$\gamma(r) = \begin{cases} 0, & \text{if } r < \alpha \\ 1, & \text{if } r > \beta \\ \frac{r - \alpha}{\beta - \alpha}, & \text{otherwise} \end{cases}$$

where $r = r(d)$ is the ratio from Equation 10.

**What $\gamma(r)$ represents.** $\gamma(r)$ is the **fraction of the original (non-interpolated) frequency to retain**. When $\gamma = 0$, the dimension is fully interpolated (its frequency is divided by $s$, exactly as in PI). When $\gamma = 1$, the dimension is fully preserved (frequency unchanged from pretraining). When $0 < \gamma < 1$, the dimension receives a weighted combination.

**The per-dimension frequency modification.** The "NTK-by-parts" method sets $g(m) = m$ (position indices unchanged) and modifies the frequencies via:

$$h(\theta_d) = \left(1 - \gamma(r(d))\right) \frac{\theta_d}{s} + \gamma(r(d)) \theta_d$$

**What this computes.** For each dimension $d$:
- Compute the ratio $r(d) = L / \lambda_d$.
- Compute the ramp value $\gamma = \gamma(r(d))$.
- The new frequency is a weighted average of the PI-interpolated frequency $\theta_d / s$ (weight $1 - \gamma$) and the original frequency $\theta_d$ (weight $\gamma$).
- If $\gamma = 0$ (absolute-position dimension, $r(d) < 1$): the frequency becomes $\theta_d / s$ — exactly the PI interpolation, mapping all new positions back into the training range.
- If $\gamma = 1$ (relative-position dimension, $r(d) > 32$): the frequency remains $\theta_d$ — no interpolation at all, preserving the fine-grained relative-position signal.
- If $0 < \gamma < 1$: the frequency is smoothly blended between the two extremes.

**Why this specific form.** The linear ramp over the ratio $r(d)$ is a simple, interpretable choice. The paper notes that alternatives exist: "The interpolation by linear ramp on $h$ may have alternatives, such as a harmonic mean over $\theta_d/s$ and $\theta_d$ converted from a linear interpolation on wavelengths. The choice of $h$ here was for the simplicity of implementation, but both would work." The key property is not the precise functional form of the blend but the existence of distinct regimes — fully interpolated, fully preserved, and a smooth transition — which captures the wavelength-based analysis.

**How this differs from NTK-aware interpolation.** NTK-aware changes the frequency base $b$, which produces a continuous deformation of all frequencies that happens to stretch low frequencies more than high frequencies. This is an *implicit*, approximate version of the wavelength-based strategy. The problem: there is no clean separation between "interpolated" and "preserved" dimensions; some dimensions end up slightly extrapolated (their effective frequency suggests positions beyond training), making the true extension factor unpredictable. NTK-by-parts eliminates this ambiguity by making the interpolation decision explicit per dimension based on a directly interpretable criterion ($r(d)$ relative to $\alpha$ and $\beta$).

**The parameters $\alpha = 1$ and $\beta = 32$.** The choice of $\alpha = 1$ is principled: $r(d) = 1$ corresponds to the dimension whose wavelength exactly equals the pretrained context length $L$. Dimensions with wavelength longer than $L$ ($r(d) < 1$) complete less than one rotation during training and thus primarily encode absolute position — these are the dimensions that need interpolation. Dimensions with wavelength shorter than $L$ ($r(d) > 1$) complete more than one rotation — these have both relative and absolute components, and the transition to "pure relative" happens gradually. The choice of $\beta = 32$ means that dimensions completing more than 32 full rotations during training are considered "pure relative" and left untouched. This threshold was determined experimentally for LLaMA-family models and may need tuning for other architectures.

---

#### The YaRN Attention Temperature Scaling

The second component of YaRN (beyond NTK-by-parts interpolation) is a modification to the attention softmax temperature. The paper observes empirically that introducing a temperature factor $t$ on the logits before the softmax has a **uniform beneficial impact on perplexity** regardless of token position within the extended context window (documented in Appendix A.3, Figures 4–6).

**The modified attention computation.** Instead of the standard attention weight computation (Equation 2), YaRN uses:

$$\text{softmax}\left(\frac{\mathbf{q}_m^T \mathbf{k}_n}{t \sqrt{|D|}}\right)$$

where $t > 1$ is a temperature that depends on the extension scale $s$. Larger $t$ makes the softmax "softer" (more uniform attention distribution), which the paper finds necessary to stabilize perplexity when processing sequences beyond the pretrained length.

**Implementation via embedding scaling (the "length scaling" trick).** Rather than modifying the attention softmax directly (which would require changes to the attention kernel), the paper exploits a mathematical equivalence: dividing the logits by $t$ is equivalent to multiplying both query and key vectors by $\sqrt{1/t}$ before computing the dot product, because:

$$\frac{\mathbf{q}_m^T \mathbf{k}_n}{t} = \left(\sqrt{\frac{1}{t}} \mathbf{q}_m\right)^T \left(\sqrt{\frac{1}{t}} \mathbf{k}_n\right)$$

Since RoPE applies rotary embeddings to the query and key vectors, the paper scales the rotary embeddings themselves by $\sqrt{1/t}$. This is a **zero-cost operation**: the rotary embeddings are precomputed once per context length and cached; scaling them at generation time adds no per-token compute overhead. The paper emphasizes this advantage: "it has zero overhead during both inference and training, as rotary position embeddings are generated in advance and are reused for all forward passes."

**The empirical formula for $t$.** The paper determines the optimal temperature by sweeping $\sqrt{1/t}$ values on LLaMA 7B, 13B, 33B, and 65B models across different extension scales $s$ using the NTK-by-parts interpolation without fine-tuning. The result is a fitted relationship:

$$\sqrt{\frac{1}{t}} = 0.1 \ln(s) + 1$$

or equivalently:

$$t = \frac{1}{(0.1 \ln(s) + 1)^2}$$

where $s = L'/L$ is the extension scale factor.

**What this computes.** Given a target extension scale $s$, compute the natural logarithm of $s$, multiply by 0.1, add 1, and that gives $\sqrt{1/t}$ — the factor by which rotary embeddings are scaled. For example:
- $s = 2$ (doubling context): $\sqrt{1/t} = 0.1 \ln(2) + 1 \approx 0.0693 + 1 = 1.0693$ — a 7% increase in embedding magnitude.
- $s = 16$: $\sqrt{1/t} = 0.1 \ln(16) + 1 \approx 0.1 \cdot 2.773 + 1 = 1.277$ — a 28% increase.
- $s = 32$: $\sqrt{1/t} = 0.1 \ln(32) + 1 \approx 0.1 \cdot 3.466 + 1 = 1.347$ — a 35% increase.

**Why this specific form.** The logarithmic dependence on $s$ reflects that the need for softer attention grows sublinearly with the extension factor. The coefficients (0.1 and 1) were found by fitting across four model sizes (7B through 65B) and both LLaMA and Llama 2 families — the paper notes this "suggests that the property of increased entropy and the temperature constant $t$ may have certain degree of 'universality' and may be generalizable across some models and training data." This is an empirical finding, not a theoretical derivation, but its consistency across model scales is striking and practically useful.

**Why temperature scaling helps.** The paper does not provide a deep theoretical analysis, but the mechanism is interpretable. When processing sequences beyond the pretrained length, the model encounters token pairs with relative distances it has never seen. The attention logits for these unfamiliar distances may be miscalibrated — either overconfident (too peaked softmax, attending too strongly to specific tokens based on unreliable signals) or underconfident (too flat softmax, failing to focus). Increasing the temperature (making the softmax more uniform) acts as a regularizer: it prevents the model from over-committing to attention patterns that are unreliable at extended lengths. The logarithmic scaling with $s$ means this regularization grows slowly as the extension factor increases, which is consistent with the intuition that uncertainty grows gradually, not linearly, with context length.

**Evidence from Appendix A.3.** The paper validates the temperature formula extensively. Figure 4 shows that at $s = 8$, the optimal $\sqrt{1/t}$ is approximately 1.21 across 896 16k-token documents from RedPajama, with the curve showing a clear minimum in perplexity. Figure 5 demonstrates that this optimal value is consistent across different segments of the 16k-token range (from 0–2k through 14k–16k), meaning the temperature adjustment benefits all positions uniformly, not just the far end of the extended context. Figure 6 confirms that for the majority of samples, the same $\sqrt{1/t}$ value achieves the minimal perplexity regardless of position within the document. This uniformity is the key empirical property that makes the temperature scaling a simple, global adjustment rather than a position-dependent one.

**Definition of the full YaRN method.** The paper defines YaRN as:

> "a combination of the attention scaling in Eq. 14 and the 'NTK-by-parts' interpolation introduced in Section 3.2."

That is: take the NTK-by-parts frequency modification $h(\theta_d)$ (Equation 13), then additionally scale all rotary embeddings by $\sqrt{1/t}$ where $t$ follows the logarithmic fit (Equation 15). Both modifications are applied at the embedding level, requiring no changes to attention computation or model architecture.

---

#### Dynamic Scaling: Inference-Time Adaptation

The interpolation methods described so far assume a fixed extension scale $s = L'/L$, where $L'$ is the target extended context length. This works well when all sequences are approximately the same length, but fails in autoregressive generation where sequence lengths grow incrementally from 1 to potentially $L'$ or beyond. The paper identifies two problems with fixed-$s$ inference:

1. At sequence lengths $l' < L$ (shorter than the original pretrained context), the model may experience a **performance discount** because the interpolation is compressing positions that didn't need compression — the model was already trained on these lengths.
2. At sequence lengths $l' > L'$ (beyond the extended context), the model experiences an **abrupt degradation** because positions map to angles beyond what the interpolation was designed for.

**The Dynamic Scaling solution.** Instead of fixing $s$, Dynamic Scaling recomputes it on every forward pass based on the current sequence length:

$$s = \max\left(1, \frac{l'}{L}\right)$$

where $l'$ is the current sequence length and $L$ is the pretrained context length.

**What this does operationally.** At each generation step:
- If $l' \leq L$: $s = 1$ — no interpolation is applied. The model uses its original RoPE frequencies, exactly as during pretraining, so there is no performance degradation on in-distribution lengths.
- If $l' > L$: $s = l'/L$ — the interpolation factor grows linearly with the current sequence length. The model uses its extension method (NTK-by-parts, YaRN, or any other) with this dynamically computed $s$.
- The interpolation is smooth: at $l' = L$, $s$ transitions continuously from 1 to $L/L = 1$ (no discontinuity), and for $l' > L$, $s$ grows gradually.

**Why this prevents performance cliffs.** With fixed-$s$ inference, the model interpolates even at short lengths (hurting performance where it shouldn't) and fails abruptly at lengths beyond $L'$ (since $s$ was chosen for $L'$, not beyond). Dynamic Scaling eliminates both problems:
- Short sequences ($l' \leq L$) use the native RoPE with no degradation.
- Sequences between $L$ and $L'$ use progressively increasing interpolation, so the model sees a smooth transition rather than a single large jump.
- Sequences beyond $L'$ can be handled — the scale factor simply continues to grow, allowing the model to attempt lengths beyond what it was fine-tuned for, with graceful degradation rather than catastrophic failure.

**The interaction with KV-caching.** The paper identifies an important implementation detail. In autoregressive generation, previous key and value vectors are cached in the KV-cache to avoid recomputation. However, with Dynamic Scaling, the rotary position embedding for a given token changes when $s$ changes (since $s$ is updated every forward pass). The paper provides the correct implementation recipe:

> "The correct implementation should cache the kv-embeddings before applying rotary position embeddings, as the RoPE of every token changes when $s$ changes."

In other words: store the pre-RoPE key and value vectors in the cache, and apply the (dynamically scaled) rotary embeddings on-the-fly during each forward pass. This ensures that cached keys from earlier tokens use the correct (current) scale factor, not a stale one from when they were first computed.

**Dynamic NTK and Dynamic YaRN.** The Dynamic Scaling technique is independent of the specific interpolation method. When combined with NTK-aware interpolation, the paper calls it "Dynamic NTK" interpolation (first proposed by emozilla, 2023). When combined with YaRN (NTK-by-parts + temperature scaling), it becomes Dynamic-YaRN. The distinction is purely at the interpolation layer — the dynamic $s$ computation is the same.

**Performance without fine-tuning.** Dynamic Scaling is particularly effective for models that have **not** been fine-tuned for long contexts. The paper's Figure 8 (Appendix B.7) shows Dynamic-YaRN applied to a base Llama 2 model (pretrained on 4k context) without any fine-tuning: perplexity remains stable well beyond 4k tokens, whereas the baseline RoPE (no scaling) and Dynamic-PI both show sharp perplexity increases past the pretrained length. Dynamic-YaRN achieves the best long-range perplexity among the non-fine-tuned methods, demonstrating that even without any additional training, the combination of wavelength-aware interpolation and temperature scaling provides meaningful context extension.

---

#### Training Procedure

The paper's training follows the methodology established by Chen et al. (2023) for PI, with modifications to accommodate the larger extension factors YaRN enables. The training is a standard autoregressive language modeling fine-tune on long-context documents, with the only architectural change being the modified RoPE frequency computation (the YaRN equations from Sections 3.2 and 3.3).

**Dataset.** All fine-tuning uses the **PG19 dataset** (Rae et al., 2020), a collection of full-length books from Project Gutenberg published before 1919. The paper chunks this dataset into segments of the target training context length. For the 32k-context LLaMA 7B ablation experiments, segments are 32k tokens. For the 64k and 128k Llama 2 experiments, segments are 64k tokens. Each segment is bookended with BOS and EOS tokens.

**Why PG19?** The authors do not explicitly justify this choice over alternatives, but the properties are clear: PG19 contains genuinely long documents (unlike many web-scraped datasets where documents are artificially concatenated), and the pre-1919 vintage means the text is stylistically distinct from modern web text — this tests whether the context extension fine-tune overfits to a narrow distribution. The paper acknowledges in Section 4.4 that "the PG19 dataset we used for fine-tuning is very different from the original pre-training dataset used for LLaMA and Llama 2 models," yet benchmark performance is largely preserved, suggesting the extension method generalizes across distribution shifts.

**Optimization hyperparameters.** The paper uses the following configuration for all training runs:

- Optimizer: AdamW (Loshchilov and Hutter, 2019) with $\beta_1 = 0.9$ and $\beta_2 = 0.95$
- Learning rate: $2 \times 10^{-5}$
- Weight decay: none (0.0)
- Learning rate schedule: linear warmup for 20 steps, followed by constant learning rate (no decay specified)
- Global batch size: 64
- Framework: PyTorch with Fully Sharded Data Parallelism (FSDP) and Flash Attention 2

**Why these choices.** The learning rate of $2 \times 10^{-5}$ is relatively low, consistent with the goal of a "gentle" fine-tune that adapts the model to the new position encodings without overwriting pretrained knowledge. The absence of weight decay also supports this — weight decay would push parameters toward zero, potentially erasing pretrained representations, whereas the goal is minimal perturbation. The linear warmup over 20 steps is standard practice for stabilizing early training when the model encounters drastically different position encodings for the first time.

**Training length and the efficiency claim.** The paper's central efficiency claim — "10× fewer tokens and 2.5× fewer training steps than previous methods" — is anchored to this training setup:
- **YaRN (s = 16):** 400 steps at batch size 64 = 25,600 sequences. With 64k-token segments for Llama 2, this is approximately 1.64 billion tokens.
- **PI baseline (Chen et al., 2023):** The paper cites 1,000 steps as the PI training budget (Table 6 comparison). Chen et al. (2023) used billions of tokens — the "10× fewer tokens" claim compares total training tokens.
- **Table 6 provides the direct evidence:** Llama 2 7B extended from 4k to 8k. PI trained for 1,000 steps achieves perplexity of 3.34 at 8,192 context length; YaRN trained for 400 steps achieves 3.35 — essentially identical performance with 2.5× fewer steps.

**The two-phase training for s = 32 (128k context).** For the most aggressive extension (s = 32, extending Llama 2 from 4k to 128k), the paper uses a transfer learning approach due to compute constraints:
1. **Phase 1:** Fine-tune from the base Llama 2 checkpoint with s = 16 (targeting 64k context) for 400 steps on 64k-token segments. This produces the "YaRN s = 16" model.
2. **Phase 2:** Starting from the Phase 1 checkpoint, fine-tune with s = 32 for an additional 200 steps, still using 64k-token segments (not 128k segments — the training data context length remains 64k, but the interpolation factor is set to s = 32).

**Why this transfer-learning approach works.** The model never sees 128k-token sequences during training — it only sees 64k-token sequences but with position encodings interpolated by a factor of 32 (meaning the farthest token is encoded as if it were at position $64\text{k} \times 32 / 4\text{k} = 512\text{k}$ in the original pretrained space, but then mapped back via interpolation). The model learns to attend with these stretched position encodings on 64k-token sequences. At inference time, when given a 128k-token sequence, the positions are interpolated by the same factor $s = 32$, and the model generalizes — it has learned to use the interpolated encodings, not memorized a specific sequence length. The paper reports that "the s = 32 model is also trained with 64k context data, but we show that it is able to extrapolate to a context size of 128k in Section 4.2."

This is a practically significant result: training on 128k-token sequences would require substantially more GPU memory and computation than training on 64k-token sequences (roughly 2× the memory for attention). The fact that YaRN can be trained on shorter sequences and extrapolate to longer ones means the most expensive training regime can be avoided.

**Training loss curves (Figure 2).** The paper presents training loss curves for LLaMA 7B extended to 32k (s = 16) using different interpolation methods, trained for 400 steps each. YaRN's loss curve is consistently below the other methods (PI, NTK-aware, NTK-by-parts) throughout training and converges to a lower final value. The right panel of Figure 2 zooms in on steps 200–400, showing that YaRN's advantage persists rather than being a transient early-training effect. This lower loss during training translates directly into the superior perplexity at evaluation time shown in Table 5 and Figure 3.

**Computational cost (Table 4).**

- LLaMA 7B to 32k (s = 16): 128 A100 GPU-hours
- Llama 2 7B to 64k (s = 16): 256 A100 GPU-hours
- Llama 2 7B to 128k (s = 32): 256 + 128 = 384 A100 GPU-hours (Phase 1 + Phase 2)
- Comparison: Chen et al. (2023) PI to 16k (s = 8): 640 A100 GPU-hours for a 7B model

YaRN achieves 4× the extension factor (s = 32 vs. s = 8) in approximately 60% of the GPU-hours (384 vs. 640), or equivalently, achieves the same extension factor (s = 16 for YaRN to 64k vs. s = 8 for PI to 16k) in 40% of the GPU-hours (256 vs. 640) — consistent with the 2.5× training step reduction claim.

---

#### Summary of Design Choices and Their Justifications

- **NTK-by-parts wavelength-based selective interpolation** over uniform PI: preserves the relative-position encoding that PI destroys, preventing the loss of local attention resolution that causes PI to fail beyond $s = 8$.
- **Explicit ramp function** over the NTK-aware continuous base change: eliminates the need for empirical base-tuning and ensures no dimensions are accidentally extrapolated. The thresholds $\alpha = 1$ and $\beta = 32$ have direct physical interpretations (wavelength relative to context length) rather than being opaque hyperparameters.
- **Temperature scaling $\sqrt{1/t} = 0.1 \ln(s) + 1$** over no temperature adjustment: uniformly reduces perplexity across all positions in the extended window, with a parameter-free formula that transfers across model sizes and families.
- **Implementation via embedding scaling** rather than softmax modification: zero computational overhead, full compatibility with Flash Attention 2 and other optimized attention kernels, and trivial to implement by modifying only the rotary embedding generation code.
- **Dynamic Scaling with $s = \max(1, l'/L)$** over fixed $s$: eliminates performance degradation at short lengths and enables graceful degradation beyond the fine-tuned extension limit. The kv-cache implementation detail (caching pre-RoPE embeddings) is critical for correctness.
- **PG19 dataset** for fine-tuning: provides genuinely long documents and tests generalization due to its distributional distance from standard pretraining corpora.
- **AdamW with no weight decay and low learning rate ($2 \times 10^{-5}$)**: minimal perturbation of pretrained weights, consistent with the goal of adapting only the position encoding understanding without overwriting knowledge.
- **Two-phase training for s = 32** (starting from s = 16 checkpoint): leverages transfer learning to reach the largest extension factor with reduced compute, and demonstrates that the model can "train short, test long" — learning interpolated position encodings on 64k sequences and extrapolating to 128k at inference.
- **Training data context length (64k) less than inference target (128k) for s = 32**: exploits the fact that the model learns the *interpolation scheme*, not a specific sequence length, enabling efficient training that generalizes to longer sequences than seen during fine-tuning.

## 4. Key Insights and Innovations

### Innovation 1: The Wavelength Ratio as a Diagnostic Criterion for RoPE Dimension Function

The most intellectually distinctive move in this paper is not any specific interpolation formula — it's the **diagnostic insight** that the ratio $r(d) = L / \lambda_d$ tells you what a given RoPE dimension actually *does* in the trained model. Before this paper, the field treated RoPE dimensions as a uniform set of frequencies to be manipulated en masse: PI stretches them all, NTK-aware deforms them all via a base change. The dominant assumption — implicit, never stated — was that all dimensions contribute roughly equivalently to position encoding and should be treated symmetrically.

This paper's observation cuts through that assumption. If a dimension's wavelength exceeds the pretraining context length ($r(d) < 1$), the rotary embedding never completes a full cycle during training. Every absolute position maps to a unique phase. The model can — and likely does — use this dimension to encode absolute position, because the phase-to-position mapping is one-to-one within the training distribution. Conversely, if a dimension completes many cycles during training ($r(d) \gg 1$), the phase repeats many times across positions; the model cannot use it for absolute position but can use it for relative position via RoPE's built-in relative encoding ($e^{i(m-n)\theta_d}$).

This is a **conceptual reframing**, not just a technical improvement. It transforms RoPE interpolation from a signal-processing problem ("how do we stretch all frequencies optimally?") into a **functional classification problem** ("which dimensions encode what, and what treatment does each functional class require?"). The wavelength ratio $r(d)$ becomes a diagnostic that predicts how a dimension should be handled: interpolate absolute-position dimensions (they'll go out-of-distribution otherwise), preserve relative-position dimensions (interpolating them destroys the local signal they provide).

**Why this is fundamental rather than incremental.** NTK-aware interpolation already *approximated* this behavior — its base change stretches low frequencies more than high frequencies, which happens to compress absolute-position dimensions more and relative-position dimensions less. But it did so implicitly, with no clean boundary, and it required empirical tuning of the base to get the approximation right. The NTK-by-parts method (which YaRN inherits) makes the functional distinction **explicit, parameterized, and interpretable**. The thresholds $\alpha = 1$ and $\beta = 32$ have direct physical meanings: dimensions completing less than 1 rotation during training are absolute encoders, dimensions completing more than 32 rotations are pure relative encoders, and the transition between them is smooth. There's no more empirical base-sweeping — the interpolation behavior per dimension is deterministic given $L$ and the model architecture.

**Evidence anchoring.** The ablation in Table 5 supports this reframing indirectly but powerfully: NTK-by-parts (which uses this explicit functional classification) consistently outperforms NTK-aware (which approximates it) at equivalent training, particularly at larger extension scales where the approximation errors compound. The fact that the NTK-by-parts method — developed and released independently before YaRN (bloc97, 2023b) — already outperformed both PI and NTK-aware confirms that the functional classification insight, not the specific implementation details, is what drives the improvement.

**Significance beyond raw performance.** This diagnostic concept generalizes beyond the specific thresholds $\alpha = 1$, $\beta = 32$. For any RoPE-based model with a different architecture (different $|D|$, different $b$, different $L$), the same wavelength-ratio analysis applies. One can compute $r(d)$ for each dimension, examine where the transitions between functional regimes occur, and design an interpolation strategy accordingly. The paper doesn't fully explore this generalization, but the framework is there: the wavelength ratio is a **model-agnostic diagnostic** for RoPE dimension function.

---

### Innovation 2: Attention Temperature as a Context-Length-Dependent Regularizer

The second major conceptual contribution is the discovery that **modulating attention softmax temperature as a function of context extension scale uniformly improves perplexity**, and that this effect can be implemented with zero computational overhead by scaling the rotary embeddings themselves.

This is distinctive because it identifies a **previously unnoticed failure mode** of extending RoPE beyond its trained range. The standard assumption in interpolation methods (PI, NTK-aware, NTK-by-parts) is that if you correctly map positions into the training distribution, attention will work correctly. YaRN's temperature scaling reveals that this is insufficient: even with correct position mapping, the **attention logit calibration** degrades at extended lengths. The model's confidence in its attention patterns — how peaked or flat the softmax distribution is — was tuned during pretraining for a specific range of position encodings. When those encodings are stretched (even correctly, even with wavelength-aware interpolation), the relationship between query-key dot products and appropriate attention weights shifts.

The temperature factor $t > 1$ acts as a **calibration regularizer**: it makes the softmax distribution more uniform, preventing the model from over-committing to attention patterns that may be miscalibrated at extended lengths. The logarithmic scaling $\sqrt{1/t} = 0.1 \ln(s) + 1$ means this regularization grows slowly with the extension factor — uncertainty about attention calibration grows sublinearly with context length, which has an intuitive interpretation: the most severe miscalibration happens at the boundary of the training distribution, and additional extension beyond that boundary adds diminishing additional uncertainty.

**Comparison to prior work.** No previous RoPE extension method included an attention-temperature component. PI, NTK-aware, and NTK-by-parts all leave the attention softmax unchanged — they assume that fixing the position encodings is sufficient. The paper's empirical finding that temperature scaling provides a **uniform benefit across all positions** in the extended window (Appendix A.3, Figures 4–6), not just at the far end, suggests this is a genuinely distinct failure mode from position encoding, one that prior methods simply missed because their extension factors ($s \leq 8$) were too small for the miscalibration to dominate perplexity.

**The implementation insight matters.** The equivalence between softmax temperature scaling ($\text{softmax}(\mathbf{q}^T \mathbf{k} / t)$) and embedding scaling (multiplying rotary embeddings by $\sqrt{1/t}$) is mathematically straightforward, but recognizing that this enables a **zero-overhead implementation** compatible with Flash Attention 2 is a practical contribution. Approaches that modify attention computation directly (ReRoPE, LM-Infinite) sacrifice compatibility with optimized attention kernels; YaRN's embedding-scaling trick preserves it entirely. This is the kind of implementation insight that separates a method that gets cited from one that gets deployed.

**Evidence anchoring.** Figure 3 (left) shows that YaRN — which adds temperature scaling to NTK-by-parts — achieves lower perplexity than NTK-by-parts alone across all context lengths in the fine-tuned setting. Table 5 shows the same pattern in the non-fine-tuned setting: at $s = 16$, YaRN achieves 3.45 perplexity at 32k context vs. >100 for NTK-by-parts at the same scale. The temperature scaling is the only difference between these methods at equivalent interpolation settings.

**Significance beyond the specific formula.** The fitted formula $\sqrt{1/t} = 0.1 \ln(s) + 1$ is empirical and may not transfer exactly to all models. But the **concept** — that attention calibration degrades with context extension and can be compensated by a length-dependent temperature — is likely fundamental. The paper's finding that the same formula works across LLaMA 7B through 65B and Llama 2 7B through 70B suggests it captures something about how attention uncertainty scales with position encoding stretch, not just a quirk of one model. This opens a research direction: understanding theoretically why attention miscalibration occurs under position encoding extrapolation, and whether temperature scaling is the optimal correction or a first approximation.

---

### Innovation 3: Demonstration That Position Encoding Generalization Can Be Learned on Shorter Sequences

The paper's most surprising empirical finding is that YaRN models fine-tuned on 64k-token sequences with $s = 32$ interpolation can successfully process 128k-token sequences at inference time — **twice the length they were trained on**. This is not an interpolation method innovation per se, but a **demonstration about the nature of what the model learns during context extension fine-tuning**.

The standard assumption in context extension work is that the model must be trained on sequences at the target length: if you want 128k context, you train on 128k-token segments. This assumption is expensive — attention computation scales quadratically with sequence length, so training on 128k tokens costs roughly 4× the memory and compute of training on 64k tokens. The paper's two-phase training for $s = 32$ (400 steps at $s = 16$ on 64k segments, then 200 steps at $s = 32$ on the same 64k segments) deliberately violates this assumption, and the model generalizes anyway.

**Why this is conceptually significant.** It reveals that the model is not learning a specific sequence length or memorizing position-encoding patterns for the training lengths. Instead, it is learning the **interpolation scheme itself** — it adapts its internal representations to work with the stretched frequency basis, and once that adaptation occurs, it can handle any position that maps into the interpolated encoding space, regardless of whether that exact position appeared during fine-tuning. The interpolation provides a continuous mapping from original positions to extended positions; the model learns to operate on the *mapped* positions, and since the mapping is smooth, it generalizes to mapped positions it hasn't seen.

This is a **transfer learning result** with practical and theoretical implications. Practically, it means the most expensive part of long-context training (the quadratic attention cost on very long sequences) can be avoided — train on shorter sequences with aggressive interpolation, and the model extrapolates to longer ones. Theoretically, it suggests that RoPE-based transformers can achieve length generalization through a combination of appropriate position encoding modification and modest fine-tuning, without needing to see the target length during training. This is a more optimistic picture of transformer length generalization than the "catastrophic failure" narrative that motivated the work in the first place.

**Evidence anchoring.** Table 1 provides the direct evidence: the Llama 2 7B YaRN $s = 32$ model, trained only on 64k-token segments, achieves 2.37 perplexity at 128k context length (sliding window, S = 256) — only marginally higher than its 2.45 perplexity at 64k. The trend is smooth and monotonic: 3.56 at 8k, 3.04 at 16k, 2.70 at 32k, 2.45 at 64k, 2.37 at 128k. There is no cliff at the training context boundary (64k). The passkey retrieval results in Table 9 corroborate: the $s = 32$ model achieves 99.4% accuracy at 128k context despite being trained only on 64k segments, while the $s = 16$ model (trained on 64k segments, targeting 64k) achieves lower accuracy (96.3%) on its own target length — suggesting the additional 200 steps of $s = 32$ training, even without longer sequences, meaningfully improved the model's ability to use long contexts.

**Boundaries and limitations.** The paper does not test how far this extrapolation extends — does a model trained on 64k with $s = 32$ generalize to 256k? To 512k? The passkey results suggest accuracy remains high at 128k, but perplexity trends in Figure 7 show some methods (NTK-aware Code Llama) beginning to degrade beyond 100k. Whether YaRN's extrapolation has a hard limit or degrades gracefully is an open question.

---

### Innovation 4: Dynamic Scaling as Graceful Degradation Without Fine-Tuning

Dynamic Scaling ($s = \max(1, l'/L)$ per forward pass) is, on its face, a simple inference-time trick. But its conceptual significance is larger than its implementation simplicity suggests: it reframes context extension from a **fixed-target problem** (extend from $L$ to $L'$, then deploy) to a **continuous-adaptation problem** (adapt position encodings on-the-fly to whatever length the current sequence happens to be).

This reframing matters for several reasons:

**It eliminates the performance regression at short lengths.** Fixed-$s$ methods apply interpolation even when the sequence is shorter than the pretrained context — a 2k-token sequence in a model interpolated for $s = 16$ has its positions compressed by 16×, forcing the model to operate on position encodings it was never trained on (positions 0–128 in the original space, which are valid but at a different *density* than during pretraining). Dynamic Scaling sets $s = 1$ for all $l' \leq L$, using the native RoPE frequencies exactly as during pretraining. The model's short-context performance is thus **guaranteed unchanged** — there is no interpolation to degrade it.

**It enables operation beyond the fine-tuned extension limit.** A model fine-tuned with fixed $s = 16$ (targeting 64k) breaks when given a 65k-token sequence because the interpolation was designed for positions up to 64k in the extended space. Dynamic Scaling simply increases $s$ beyond 16 as the sequence grows past 64k, allowing the model to attempt lengths it wasn't explicitly fine-tuned for. The degradation is graceful (perplexity rises gradually) rather than catastrophic (perplexity explodes to >100).

**It is method-agnostic.** Dynamic Scaling works with any interpolation method — PI, NTK-aware, NTK-by-parts, YaRN. It is a meta-strategy that can be layered on top of any position encoding modification. This means improvements to the underlying interpolation method (e.g., the progression from PI to NTK-aware to NTK-by-parts to YaRN) directly improve Dynamic Scaling's performance without changing the dynamic mechanism.

**Why this is more than an engineering trick.** The paper doesn't frame it this way, but Dynamic Scaling implicitly acknowledges that **context length is not a property of the model — it's a property of the current sequence**. A model doesn't "have" a context window; it "has" position encodings that are well-calibrated for some range of lengths and progressively less calibrated beyond that range. Dynamic Scaling operationalizes this view by continuously adapting the encoding to the current sequence, letting the model operate in its best-calibrated regime at every length. This is a more nuanced view of length generalization than the binary "in-distribution vs. out-of-distribution" framing that motivated earlier work.

**Evidence anchoring.** Figure 8 (Appendix B.7) shows Dynamic-YaRN applied to a non-fine-tuned Llama 2 model: perplexity remains stable well past the 4k pretrained limit, whereas the baseline RoPE and Dynamic-PI both show sharp increases. Table 5 shows that Dynamic-YaRN without fine-tuning achieves 3.45 perplexity at 32k context (from a 2k pretrained LLaMA 7B) — competitive with some *fine-tuned* methods at lower extension scales. The method works without any additional training at all.

**Comparison to prior work.** Previous approaches treated context extension as a property of the model to be fixed at training or fine-tuning time. Dynamic NTK (emozilla, 2023) introduced the idea of per-forward-pass scale adjustment, but the paper generalizes and formalizes it, and importantly identifies the **KV-cache interaction** — caching pre-RoPE embeddings rather than post-RoPE vectors — as the critical implementation detail for correctness. This detail is non-obvious (many implementations cache post-RoPE vectors and would silently produce incorrect results with Dynamic Scaling) and demonstrates the kind of careful systems thinking that distinguishes a deployed method from a conceptual one.

**Limitations.** Dynamic Scaling does not solve the fundamental problem of position encoding extrapolation — it only makes the degradation graceful rather than catastrophic. For sequences far beyond the fine-tuned or pretrained length, perplexity will still eventually rise to unusable levels. The paper does not characterize *how* gracefully the degradation occurs for YaRN specifically, or whether there is a length at which Dynamic-YaRN also fails catastrophically. This is a significant open question for deployment: if you rely on Dynamic Scaling for variable-length inputs, you need to know at what length to truncate or fall back to a different strategy.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All fine-tuning uses the PG19 dataset (Rae et al., 2020), a collection of full-length Project Gutenberg books published before 1919, chunked into contiguous segments of the target training context length (32k tokens for LLaMA 7B ablation experiments, 64k tokens for Llama 2 experiments) and bookended with BOS and EOS tokens. Evaluation uses long-document benchmarks: Proof-pile (Azerbayev et al., 2022), from which 10 random samples with at least 128k tokens each are selected for perplexity evaluation, and GovReport (Huang et al., 2021), using 50 untruncated documents with at least 16k tokens each. Standardized benchmarks from the Hugging Face Open LLM Leaderboard (Hugging Face, 2023) — ARC-Challenge (25-shot), HellaSwag (10-shot), MMLU (5-shot), and TruthfulQA (0-shot) — assess short-context performance preservation. The passkey retrieval task (Mohtashami and Jaggi, 2023) measures a model's ability to retrieve a five-digit number from among otherwise meaningless text, with the passkey placed at a random location uniformly distributed across the evaluation context window.

- **Base model(s).** Ablation studies use LLaMA 7B (Touvron et al., 2023a), which has a pretrained context window of 2,048 tokens. Primary experiments on large extension factors use Llama 2 7B and 13B (Touvron et al., 2023b), which have pretrained context windows of 4,096 tokens. LLaMA 7B is chosen for ablations because it shares the Llama 2 architecture but has a shorter pretrained context window, reducing compute requirements for faster training and evaluation. Llama 2 models are chosen because they represent a widely-used open model family at the time of publication, and extending them to long contexts is of direct practical interest to the open-source community.

- **Metrics.** The primary evaluation metric is **sliding window perplexity** with window size $S = 256$ (Press et al., 2022), which computes perplexity over 256-token chunks and averages across all chunks in a document, ensuring that long documents contribute proportionally to their length rather than being dominated by a single aggregate perplexity score. For passkey retrieval, the metric is **accuracy** — the fraction of test instances where the model correctly outputs the hidden passkey, measured across context window sizes ranging from 2k to 128k. For standardized benchmarks, task-specific accuracy metrics are used as defined by the Hugging Face Open LLM Leaderboard. Training efficiency is measured in **A100 GPU-hours** for direct comparability across methods.

- **Baselines.** The paper compares against four interpolation methods, each representing a stage in the evolutionary lineage from Figure 1: **Position Interpolation (PI)** (Chen et al., 2023; kaiokendev, 2023), which uniformly scales all RoPE dimensions by $s$; **"NTK-aware" interpolation** (bloc97, 2023a), which changes the RoPE base frequency to $b' = b \cdot s^{|D|/(|D|-2)}$; **"NTK-by-parts" interpolation** (bloc97, 2023b), which selectively interpolates dimensions based on wavelength thresholds using the ramp function but without the attention temperature scaling; and the **unmodified base model** (no extension) as a pretraining-performance reference point. For external comparisons (Tables 7 and 9), the paper benchmarks against publicly available long-context models: **Together.ai's LLaMA-2-7B-32K** (Together.ai, 2023), which uses PI to extend Llama 2 to 32k context, and **Code Llama** (Rozière et al., 2023), which uses NTK-aware interpolation with base $b = 1,000,000$ to achieve approximately 100k context.

- **Generation budget / compute accounting.** All interpolation methods modify only the rotary position embedding computation, which is precomputed and cached, so there is **no per-token inference cost difference** between methods. Training cost is measured in three dimensions: number of training steps, total tokens processed, and A100 GPU-hours. The paper fixes training at 400 steps with global batch size 64 for all YaRN ablation experiments, and compares against PI baselines trained for 1,000 steps (Table 6). Compute accounting for the FLOPs-matched comparison with prior work is provided in Table 4, which reports total A100-hours for each method.

- **Cross-validation / statistical protocol.** The paper does not employ formal cross-validation or statistical significance testing. Perplexity evaluations on Proof-pile use 10 documents (Table 1, Table 5) or a single 1.28M-token document (Figure 7), with results reported as single-point estimates rather than confidence intervals. Passkey retrieval uses 50 iterations for the 32k LLaMA 7B evaluation (Figure 3, right) and 10 iterations for the 64k and 128k Llama 2 evaluations (Table 9), with accuracy averaged across trials. Standardized benchmark evaluations use the full test sets as defined by the Hugging Face leaderboard. The absence of confidence intervals or standard errors means that small performance differences (e.g., the 0.49% average score drop between $s = 16$ and $s = 32$ models in Section 4.4) cannot be assessed for statistical reliability.

---

### Main Quantitative Results

#### Long Sequence Language Modeling: Perplexity on Proof-pile

The headline result for context extension is that YaRN successfully extends Llama 2 models to 128k context length (a 32× increase over the pretrained 4k window) while maintaining monotonically decreasing or stable perplexity across the entire extended range. Table 1 reports sliding window perplexity ($S = 256$) on ten 128k-token Proof-pile documents:

- **Llama 2 7B, YaRN $s = 16$ (64k target):** perplexity decreases from 3.51 at 8,192 tokens to 2.42 at 65,536 tokens, before exploding to >100 at 131,072 tokens (beyond the 64k training target, as expected).
- **Llama 2 7B, YaRN $s = 32$ (128k target):** perplexity decreases from 3.56 at 8,192 tokens to 2.37 at 131,072 tokens, with a smooth monotonic trend across all lengths: 3.56 → 3.04 → 2.70 → 2.45 → 2.37 as context doubles from 8k to 128k. Critically, there is no cliff at the 64k training data boundary — the model trains on 64k-token segments with $s = 32$ and successfully extrapolates to 128k.
- **Llama 2 13B, YaRN $s = 32$:** the same smooth trend from 3.29 at 8k to 2.24 at 128k, with the 13B model consistently outperforming the 7B model at every context length by roughly 0.1–0.3 perplexity points.

The transfer learning pattern is evident: the $s = 32$ model (trained for 400 steps at $s = 16$, then 200 additional steps at $s = 32$, all on 64k-token data) achieves better perplexity at 64k (2.45 vs. 2.42 for the $s = 16$ model that was trained specifically for 64k) and extends successfully to 128k. This is the "train short, test long" result.

**Comparison across interpolation methods (Figure 3, left; Table 5).** On LLaMA 7B extended to 32k context ($s = 16$) with 400 training steps:

- **YaRN (fine-tuned):** achieves 2.77 perplexity at 32,768 tokens — the lowest among all methods.
- **NTK-by-parts (fine-tuned):** achieves 2.81 at 32k — close to YaRN but consistently slightly worse.
- **NTK-aware (fine-tuned):** achieves 8.49 at 32k — dramatically worse, confirming the paper's claim that NTK-aware interpolation performs poorly when fine-tuned.
- **PI (fine-tuned):** achieves 3.57 at 32k — better than NTK-aware but substantially worse than NTK-by-parts and YaRN.

At shorter context lengths (4,096 tokens), the fine-tuned methods cluster more tightly — YaRN at 3.77, NTK-by-parts at 3.75, NTK-aware at 3.92, PI at 4.95 — suggesting that the advantage of wavelength-aware interpolation grows with context length, as the accumulation of small per-dimension errors in PI and NTK-aware compounds over longer sequences.

**Comparison without fine-tuning (Table 5, non-fine-tuned section).** At $s = 16$ without any fine-tuning, the pattern is striking:

- **YaRN (non-fine-tuned):** 3.45 perplexity at 32k — remarkably, this is *better* than PI *with* fine-tuning (3.57).
- **NTK-by-parts (non-fine-tuned):** explodes to >100 at 16k and beyond — the method completely fails at large extension factors without fine-tuning, despite outperforming NTK-aware and PI at lower scales.
- **NTK-aware (non-fine-tuned):** 6.85 at 16k, >100 at 32k — fails at large scales.
- **PI (non-fine-tuned):** >100 at all lengths beyond 8k — fails immediately.

This reveals a critical interaction: the NTK-by-parts interpolation (without temperature scaling) works well at moderate scales ($s = 4, 8$) without fine-tuning but catastrophically fails at $s = 16$. YaRN's temperature scaling component is what enables non-fine-tuned performance at $s = 16$, and this carries over to the fine-tuned setting where YaRN consistently outperforms NTK-by-parts.

**Comparison against external models (Table 7, Figure 7).** Evaluated on a single 1.28M-token Proof-pile document with sliding window $S = 256$:

- **Llama 2 7B YaRN $s = 32$ (128k context):** outperforms all external baselines at their respective target lengths, achieving 2.37 at 128k.
- **Llama 2 7B YaRN $s = 16$ (64k context):** achieves 2.42 at 64k, compared to Together.ai's PI-based 32k model which achieves 2.64 at 32k and explodes to >100 beyond its training length.
- **Code Llama 7B (100k context, NTK-aware):** achieves 2.55 at 65,536 tokens and 2.71 at 131,072 — better than the Together.ai model and competitive with YaRN at 64k, but with slightly higher perplexity at its maximum context and a gradual upward trend beyond 100k (2.55 → 2.54 → 2.71 from 65k to 131k) that suggests degradation, unlike YaRN's monotonically decreasing trend.
- **Code Llama 13B (100k context):** similar pattern — 2.41 at 65k, 2.37 at 98k, 2.54 at 131k. YaRN 13B $s = 32$ achieves 2.31 at 65k and 2.24 at 128k — consistently lower.

The visualized trends in Figure 7 make the comparison stark: Together.ai's model shows a sharp perplexity cliff beyond 32k (exploding to >10^4 by 100k), while both Code Llama and YaRN maintain reasonable perplexity to 128k, with YaRN showing the most stable and lowest trajectory.

#### Long Sequence Language Modeling: Perplexity on GovReport

Table 8 reports sliding window perplexity ($S = 256$) on 50 GovReport documents with a fixed context window of 32k:

- **Llama 2 7B YaRN $s = 16$:** 3.59 — best among 7B models.
- **Together.ai 7B (PI, 32k):** 3.67 — slightly worse.
- **Code Llama 7B (NTK-aware, 100k):** 4.44 — substantially worse, despite having more context capacity.
- **Llama 2 13B YaRN $s = 16$:** 3.35 — best overall, with the 13B model scaling as expected.

The 32k evaluation length is within the training context for all models (Together.ai trained to 32k, YaRN $s = 16$ to 64k, Code Llama to 16k training data), so this measures in-distribution long-context modeling quality rather than extrapolation. YaRN's advantage on GovReport is consistent with the Proof-pile results.

#### Passkey Retrieval

The passkey retrieval task measures a fundamentally different capability than perplexity: can the model actually *use* information from arbitrary positions within the extended context, or does it only achieve low perplexity by learning to ignore distant tokens?

**LLaMA 7B fine-tuned to 32k (Figure 3, right).** With 50 trials per context length, passkey placed uniformly at random:

- **YaRN:** maintains near-100% accuracy through 24k, drops slightly to approximately 95% at 32k. The curve is the highest among all methods at all context lengths.
- **NTK-by-parts:** broadly similar but slightly lower — approximately 90% at 32k.
- **NTK-aware:** shows a sharp drop starting around 16k, falling to approximately 40% at 24k and near 0% at 32k — confirming that NTK-aware fine-tuning fails to teach the model to attend across the full extended context even when perplexity is reasonable.
- **PI:** drops precipitously after 8k, reaching near 0% by 20k.

This is one of the paper's most discriminating results: NTK-aware achieves moderate perplexity at 32k (Table 5) but cannot retrieve a passkey — the model has learned to produce fluent text at long contexts without actually attending to distant tokens. YaRN's superior passkey performance indicates that its lower perplexity reflects genuine long-range attention rather than an artifact of local-context prediction.

**Llama 2 7B and 13B, YaRN $s = 16$ and $s = 32$ (Table 9).** With 10 trials per context length:

- **Llama 2 7B YaRN $s = 32$ (128k):** 99.4% average accuracy across all tested context sizes up to 128k — essentially perfect retrieval across the entire extended window.
- **Llama 2 7B YaRN $s = 16$ (64k):** 96.3% average accuracy up to 64k — slightly lower than the $s = 32$ model, despite being the model specifically trained for 64k. The paper hypothesizes this model "might be relatively undertrained for the passkey retrieval task."
- **Llama 2 13B YaRN $s = 32$:** 99.4% — identical to the 7B at this ceiling.
- **Code Llama 13B (100k context, NTK-aware):** 99.4% accuracy up to 128k. Despite its perplexity beginning to degrade beyond 100k (Table 7: 2.54 at 131k vs. 2.37 at 98k), passkey retrieval remains strong — the paper notes this suggests "perplexity may not be a great indicator of whether an LLM is able to attend to all tokens."
- **Code Llama 7B:** 94.3% accuracy up to 112k — slightly lower than the 13B variant and YaRN models.
- **Together.ai 7B (PI, 32k):** 100% up to 32k (its training length) but not evaluated beyond.

The key pattern: YaRN models retrieve passkeys accurately across their entire extended context window, including beyond the training data length (64k → 128k for $s = 32$). The $s = 32$ model's marginally better retrieval than $s = 16$ on the same training data suggests that the additional 200 steps of $s = 32$ training, even without longer sequences, meaningfully improved the model's ability to use long-range information.

#### Preserving Short-Context Performance on Standardized Benchmarks

A critical requirement for context extension methods is that they do not degrade the model's original capabilities on short sequences. Table 2 reports LLaMA 7B (2k pretrained) extended to 32k ($s = 16$) with 400 training steps, evaluated on the Hugging Face Open LLM Leaderboard:

- **Base LLaMA 7B (no extension):** ARC-c 51.0, HellaSwag 77.8, MMLU 35.7, TruthfulQA 34.3.
- **YaRN $s = 16$:** 48.1, 77.2, 30.0, 35.1 — minimal degradation on ARC-c (−2.9 points), HellaSwag (−0.6), and TruthfulQA (+0.8), with a larger drop on MMLU (−5.7).
- **NTK-by-parts $s = 16$:** 48.5, 76.6, 32.7, 33.4 — comparable to YaRN, slightly better on MMLU.
- **NTK-aware $s = 16$:** 47.4, 73.9, 27.7, 32.6 — notably worse on HellaSwag (−3.9 vs. base) and MMLU (−8.0).
- **PI $s = 16$:** 44.8, 70.2, 25.9, 34.1 — worst degradation across the board, losing 6.2 points on ARC-c, 7.6 on HellaSwag, and 9.8 on MMLU.

The pattern is clear: wavelength-aware methods (NTK-by-parts, YaRN) preserve short-context capabilities better than uniform interpolation (PI) or continuous base-change methods (NTK-aware). The MMLU degradation is the largest for all methods, which the paper attributes to the distributional mismatch between the PG19 fine-tuning data (pre-1919 books) and the original LLaMA pretraining data.

Table 3 reports Llama 2 7B and 13B YaRN models ($s = 16$ and $s = 32$) on the same benchmarks, compared to the base Llama 2 models (4k pretrained):

- **Llama 2 7B base:** ARC-c 53.1, HellaSwag 77.8, MMLU 43.8, TruthfulQA 39.0.
- **7B YaRN $s = 16$:** 52.3, 78.8, 42.5, 38.2 — small drops on ARC-c (−0.8), MMLU (−1.3), and TruthfulQA (−0.8), with a slight *improvement* on HellaSwag (+1.0).
- **7B YaRN $s = 32$:** 52.1, 78.4, 41.7, 37.3 — further small drops, with the paper noting "on average a 0.49% drop in scores between the YaRN $s = 16$ and $s = 32$ models."
- **Llama 2 13B base:** 59.4, 82.1, 55.8, 37.4.
- **13B YaRN $s = 16$:** 58.1, 82.3, 52.8, 37.8 — drops on ARC-c (−1.3) and MMLU (−3.0), slight improvements on HellaSwag (+0.2) and TruthfulQA (+0.4).
- **13B YaRN $s = 32$:** 58.0, 82.2, 51.9, 37.3 — further small drops, with MMLU showing the largest degradation (−3.9 from base).

The key takeaway: extending the context window by 16–32× using YaRN costs roughly 1–3 points on most benchmarks, with MMLU (a knowledge-intensive task) showing the largest drops likely due to the PG19 fine-tuning distribution shift rather than the position encoding modification itself. The iterative extension from $s = 16$ to $s = 32$ incurs an additional ~0.5% degradation on average — negligible given the doubling of context length.

Table 10 provides the full comparison including external models:

- **Together.ai 7B (PI, 32k):** 47.6, 76.1, 43.3, 39.2 — ARC-c drops 5.5 points from the Llama 2 7B base, substantially worse than YaRN's 0.8-point drop.
- **Code Llama 7B (NTK-aware, 100k):** 39.9, 60.8, 31.1, 37.8 — devastating drops on ARC-c (−13.2), HellaSwag (−17.0), and MMLU (−12.7), likely due to Code Llama being primarily trained on code rather than natural language, not due to the NTK-aware extension method per se. The paper includes this for completeness but the comparison is confounded by training data differences.
- **Code Llama 13B:** similarly large drops (40.9, 63.4, 32.8, 43.8) compared to Llama 2 13B base.

#### Training Efficiency

Table 4 provides the computational cost comparison in A100 GPU-hours:

- **LLaMA 7B YaRN to 32k ($s = 16$):** 128 hours.
- **Llama 2 7B YaRN to 64k ($s = 16$):** 256 hours.
- **Llama 2 7B YaRN to 128k ($s = 32$):** 384 hours (256 + 128 for the two-phase training).
- **Chen et al. (2023) PI to 16k ($s = 8$):** 640 hours — for a *smaller* extension factor.
- **Together.ai (PI to 32k):** cost not reported ("?").
- **Xiong et al. (2023) (NTK-aware, ~50k):** 64,000 hours — a fully from-scratch pretraining extension approach that is two orders of magnitude more expensive.
- **Code Llama (NTK-aware, ~100k):** 6,400 hours — substantially more expensive than YaRN's 384 hours for 128k.

The efficiency claim — "10× fewer tokens and 2.5× fewer training steps than previous methods" — is substantiated by the 400-step YaRN training vs. 1,000-step PI training for comparable results (Table 6), and the 256 A100-hours for YaRN to 64k vs. 640 hours for PI to 16k. The 10× token reduction follows from the shorter training and smaller context length used during training (64k vs. the full target length that other methods require).

Table 6 provides the direct side-by-side evidence for the 2.5× claim: Llama 2 7B extended from 4k to 8k ($s = 2$). PI trained for 1,000 steps achieves 3.34 perplexity at 8,192 tokens; YaRN trained for 400 steps achieves 3.35 — "essentially identical performance with 2.5× fewer steps." Non-fine-tuned YaRN at this scale achieves 3.49 — better than non-fine-tuned PI (3.65) and close to the fine-tuned PI result, demonstrating that even without training, YaRN outperforms untrained PI and approaches trained PI at modest extension factors.

---

### Ablation Studies and Robustness Checks

**Training loss curves across interpolation methods (Figure 2):** LLaMA 7B fine-tuned to 32k for 400 steps. YaRN's training loss is consistently below PI, NTK-aware, and NTK-by-parts throughout the entire training run. The right panel zooms in on steps 200–400, showing that the gap persists rather than closing — YaRN converges to a lower final loss, and the ordering (YaRN < NTK-by-parts < NTK-aware < PI) matches the evaluation perplexity ordering. This confirms that the advantage is not a function of training duration and would persist with additional steps.

**Non-fine-tuned performance across extension scales (Table 5, non-fine-tuned section):** This is effectively an ablation of the temperature scaling component. Without fine-tuning:
- At $s = 4$: YaRN (3.65 at 8k) outperforms NTK-by-parts (4.11) and NTK-aware (>100), while PI already fails (>100 at 8k).
- At $s = 8$: YaRN (3.33 at 16k) still performs well, while NTK-by-parts degrades (5.79 at 16k) and NTK-aware fails (>100).
- At $s = 16$: YaRN (3.45 at 32k) remains functional and *outperforms fine-tuned PI* (3.57), while NTK-by-parts, NTK-aware, and PI all produce >100 perplexity.

This isolates the temperature scaling as the critical component enabling non-fine-tuned performance at large extension factors. NTK-by-parts (without temperature scaling) works at $s = 2, 4$ but fails catastrophically at $s = 16$; YaRN (NTK-by-parts + temperature scaling) works across all tested scales. The fact that non-fine-tuned YaRN at $s = 16$ matches or exceeds fine-tuned PI is one of the paper's most surprising results.

**Attention temperature sweep (Appendix A.3, Figures 4–6):** On 896 16k-token RedPajama documents with LLaMA 7B at $s = 8$:
- Figure 4: sweeping $\sqrt{1/t}$ from 1.0 to 1.6, the optimal value is approximately 1.21, matching the formula $\sqrt{1/t} = 0.1 \ln(8) + 1 \approx 1.208$. The perplexity curve shows a clear minimum, confirming that the temperature is not merely "higher is better" but has an optimal value.
- Figure 5: the same optimal temperature provides the best perplexity consistently across all position segments from 0–2k through 14k–16k, demonstrating that the benefit is uniform across the extended context, not limited to the far end.
- Figure 6: counting which $\sqrt{1/t}$ value achieves minimal perplexity for each of 896 samples across 8 position segments, the mode and distribution are centered near the formula's prediction, with the majority of samples benefiting from the same value. This uniformity justifies the use of a single global temperature rather than a position-dependent one.

**Wavelength threshold sensitivity ($\alpha = 1$, $\beta = 32$):** The paper does not perform a formal sensitivity analysis or sweep over $\alpha$ and $\beta$ values. It states that "for the Llama family of models, good values for $\alpha$ and $\beta$ are $\alpha = 1$ and $\beta = 32$" based on experimental findings, but no table or figure demonstrates the impact of different threshold choices. This is a notable gap — the reader cannot assess how sensitive the method is to these parameters or whether the same values would transfer to non-LLaMA architectures.

**Two-phase training vs. direct $s = 32$ training:** The paper does not provide an ablation comparing the two-phase approach (400 steps at $s = 16$, then 200 steps at $s = 32$) against training directly at $s = 32$ for 600 steps from the base checkpoint. The two-phase approach is adopted due to compute constraints, not because it was shown to be superior. Whether the $s = 16$ intermediate step is necessary or merely convenient is unknown.

**Training data context length for $s = 32$:** The $s = 32$ model is trained on 64k-token segments, not 128k. The paper demonstrates that this works — the model extrapolates to 128k at inference — but does not compare against training on 128k segments. This is partially a compute constraint (128k training would require more memory), but an ablation comparing 64k and 128k training data for the same $s = 32$ target would clarify whether the shorter training sequences impose any performance penalty that could be recovered with longer training data.

**Interpolation method on standardized benchmarks (Table 2):** The comparison across PI, NTK-aware, NTK-by-parts, and YaRN on short-context benchmarks doubles as an ablation of the interpolation strategy's impact on preserved capabilities. NTK-by-parts and YaRN preserve ARC-c and HellaSwag substantially better than PI (48.1 and 48.5 vs. 44.8 on ARC-c; 77.2 and 76.6 vs. 70.2 on HellaSwag), confirming that wavelength-aware interpolation causes less disruption to the model's internal representations. The MMLU drop is large for all methods (base 35.7 → 25.9–32.7), suggesting a primary effect of the PG19 fine-tuning distribution rather than the interpolation method.

**Dynamic Scaling on non-fine-tuned models (Appendix B.7, Figure 8):** Applying Dynamic-YaRN to a base Llama 2 model without any fine-tuning shows that perplexity on a long GovReport document remains stable well past the 4k pretrained limit, whereas the baseline RoPE (no modification) and Dynamic-PI both show sharp perplexity increases. This demonstrates that Dynamic Scaling is not merely a supplement to fine-tuning — it provides meaningful context extension on its own, and Dynamic-YaRN outperforms Dynamic-PI.

**Dynamic Scaling method comparison (Table 5, "Dynamic" rows):** In the non-fine-tuned setting, applying Dynamic versions of each method at evaluation time (varying $s$ per forward pass based on current sequence length) shows that Dynamic-YaRN matches the performance of fixed-$s$ YaRN at each target scale — 3.45 at 32k for Dynamic-YaRN vs. 3.45 for fixed $s = 16$ — while also handling shorter sequences without degradation. Dynamic-NTK-aware and Dynamic-NTK-by-parts similarly match their fixed-scale counterparts. This confirms that Dynamic Scaling loses no performance compared to fixed-scale inference while gaining the ability to handle variable-length sequences gracefully.

**Negative result: NTK-aware fine-tuning degradation (Table 5, fine-tuned section):** While NTK-aware interpolation outperforms PI without fine-tuning, it degrades severely when fine-tuned — 8.49 perplexity at 32k vs. 2.81 for NTK-by-parts and 2.77 for YaRN. The paper attributes this to the fact that NTK-aware's base change causes some dimensions to be extrapolated (mapped to frequencies corresponding to positions beyond the pretrained range), and fine-tuning on these "out-of-bound" values causes the model to overfit to incorrect position representations. This negative result is important because it explains why Code Llama (which uses NTK-aware) requires orders of magnitude more training compute — it is fighting against an interpolation scheme that is fundamentally incompatible with fine-tuning.

**Negative result: PI fails beyond $s = 8$ even with fine-tuning (Table 5):** PI with 400 training steps at $s = 16$ achieves only 3.57 at 32k context — better than NTK-aware but still substantially worse than NTK-by-parts (2.81) and YaRN (2.77), and the non-fine-tuned PI at $s = 16$ produces >100 perplexity. This empirically anchors the paper's claim that uniform interpolation has an inherent scaling limit that wavelength-aware methods overcome.

---

### Critical Assessment

#### Claim 1: "YaRN requires 10× fewer tokens and 2.5× fewer training steps than previous methods."

The 2.5× training step reduction is directly supported by Table 6: at $s = 2$ (4k → 8k), YaRN in 400 steps matches PI in 1,000 steps (3.35 vs. 3.34 perplexity). This is a clean, controlled comparison — same model, same dataset, same target length, same evaluation metric, different interpolation methods. The evidence is strong for this specific extension factor and model combination.

However, the "10× fewer tokens" claim is less directly substantiated. Table 4 reports that Chen et al. (2023) used 640 A100-hours for PI to 16k ($s = 8$) vs. 256 A100-hours for YaRN to 64k ($s = 16$). This is a factor of 2.5× in GPU-hours for a 2× larger extension factor, not a direct token comparison. The 10× figure likely derives from the combination of 2.5× fewer steps, shorter training sequences (64k for YaRN $s = 16$ vs. presumably 16k for PI $s = 8$, which is 4× fewer tokens per sequence), and the fact that PI methods typically train on the full target context length while YaRN can train on shorter sequences. But the paper does not provide a line-item breakdown of total tokens processed for each method, so the 10× claim is plausible but not rigorously demonstrated with the reported data. The PI training data details (context length used, total tokens) from Chen et al. (2023) are not reproduced, making independent verification difficult.

Additionally, the 2.5× claim is demonstrated only at $s = 2$ (Table 6), not at the more aggressive extension factors ($s = 16$, $s = 32$) where YaRN's advantages are most pronounced. At larger $s$, PI degrades to the point where no amount of additional training would match YaRN's performance — the efficiency comparison becomes undefined because the methods don't converge to the same performance ceiling. The training step comparison is therefore most meaningful at modest extension factors where both methods can reach comparable accuracy, and becomes less interpretable at the scales ($s = 16, 32$) that are YaRN's primary claim to novelty.

#### Claim 2: "YaRN successfully extends Llama 2 models to 128k context length — a 32× increase."

Strongly supported by Table 1 (perplexity decreases monotonically from 8k to 128k for the $s = 32$ model), Table 7 (YaRN 128k achieves the lowest perplexity among all compared models at 128k), and Table 9 (99.4% passkey retrieval accuracy at 128k). The evidence spans three complementary evaluation modalities — perplexity, retrieval, and benchmark preservation — and all three paint a consistent picture.

A qualification: the 128k context model was trained on 64k-token segments. The passkey accuracy is 99.4%, but the trials use only 10 iterations per context length (Table 9). For the 32k LLaMA 7B evaluation, 50 iterations were used (Figure 3). The reduction to 10 iterations for the larger models may have been a compute constraint, but it means the passkey accuracy estimates have wide confidence intervals — a single failure drops accuracy by 10 percentage points. The 99.4% figure should be interpreted as "high but not perfectly measured."

#### Claim 3: "YaRN preserves original short-context benchmark performance with minimal degradation."

Supported with qualifications. Table 3 shows that Llama 2 7B YaRN $s = 16$ loses 0.8 points on ARC-c, gains 1.0 on HellaSwag, loses 1.3 on MMLU, and loses 0.8 on TruthfulQA — small changes that are arguably within the noise of benchmark evaluation (no confidence intervals are reported). The 13B model shows slightly larger drops (ARC-c −1.3, MMLU −3.0) but HellaSwag and TruthfulQA are essentially flat.

However, three caveats temper this claim:

1. **MMLU degradation is consistently the largest across all methods and model sizes** (Tables 2, 3, 10). The paper attributes this to the PG19 fine-tuning dataset being distributionally different from the original pretraining data. This means the degradation is not necessarily a consequence of the position encoding modification — it may be catastrophic forgetting from fine-tuning on out-of-distribution text. An ablation fine-tuning the base model on PG19 *without* changing position encodings would isolate the two effects, but this control experiment is not reported. The claim that YaRN "preserves" benchmark performance conflates preservation of capabilities through the interpolation method with preservation through the fine-tuning process — these are distinct mechanisms, and the paper's evidence does not separate them.

2. **The ARC-c score for LLaMA 7B YaRN (48.1) is lower than both NTK-by-parts (48.5) and the base model (51.0).** The paper presents this as comparable performance, but the base-to-YaRN drop of 2.9 points on ARC-c is non-trivial for a 51-point baseline (~6% relative). Whether this is "minimal" depends on the use case — for some applications, a 6% drop on reasoning benchmarks would be unacceptable.

3. **Only four benchmarks are tested.** The Hugging Face leaderboard is a standard suite but far from comprehensive. Capabilities not captured by these four tasks — instruction following, coding, mathematical reasoning, multilingual performance — are not evaluated. The claim of "minimal degradation" is scope-limited to the tested benchmarks.

#### Claim 4: "Dynamic-YaRN allows for more than 2× context window extension without any fine-tuning."

Supported by Figure 8 (Appendix B.7), which shows Dynamic-YaRN maintaining stable perplexity on a GovReport document well past the Llama 2 4k pretrained limit, while RoPE and Dynamic-PI degrade. The experiment uses a single document, making the result suggestive rather than conclusive — a 50-document evaluation like the GovReport results in Table 8 would be more convincing.

Table 5 provides additional support: non-fine-tuned Dynamic-YaRN achieves 3.45 perplexity at 32k on Proof-pile for LLaMA 7B ($s = 16$, a 16× extension from 2k pretrained), matching fixed-$s$ non-fine-tuned YaRN and outperforming fine-tuned PI. This is a stronger result — it uses multiple documents and a standard evaluation protocol.

However, the claim of "more than 2×" is oddly conservative given the evidence. Dynamic-YaRN without fine-tuning achieves $s = 16$ (16× extension) in Table 5 with usable perplexity (3.45) — far more than 2×. The "2×" figure may reflect a worst-case or a specific model/evaluation combination, but the paper's own data supports much larger non-fine-tuned extensions. The discrepancy between the stated claim and the demonstrated capability is not explained.

#### Strengths of the Experimental Design

**Multi-faceted evaluation.** The paper tests context extension along three complementary dimensions — perplexity (does the model produce fluent text?), passkey retrieval (does the model actually attend to distant tokens?), and standardized benchmarks (are original capabilities preserved?) — each addressing a distinct failure mode. A method could achieve low perplexity by learning to ignore distant context (which would fail passkey retrieval) or could achieve long contexts at the cost of short-context regression (which would fail benchmarks). YaRN passes all three tests, and the NTK-aware failure on passkey retrieval despite reasonable perplexity (Figure 3 right, Table 5) validates the importance of testing all three.

**Fair training budget comparisons.** All ablation experiments (Figure 2, Figure 3, Table 2, Table 5 fine-tuned section) use identical training steps (400), batch size (64), dataset (PG19), and hyperparameters, varying only the interpolation method. This isolates the effect of the interpolation strategy from confounding factors like training duration or data quantity. The comparison against PI at 1,000 steps (Table 6) acknowledges that PI may need more training and tests whether YaRN at 400 steps can match PI at 1,000 — it can, strengthening the efficiency claim.

**Transfer learning demonstration.** The two-phase $s = 32$ training (starting from the $s = 16$ checkpoint) and the evaluation on 128k despite 64k training data (Table 1) is a genuine demonstration of generalization — the model processes sequences twice as long as any it saw during fine-tuning. This is not an incremental improvement over prior work but a qualitatively different result: previous methods required training at the target context length.

**External model comparisons.** Tables 7, 8, 9, and 10 include publicly available models (Together.ai, Code Llama) evaluated under the same protocol, providing external validity beyond the paper's own training runs. The comparisons reveal that YaRN's advantage is not merely over its own ablation baselines but over independently developed and trained models.

#### Weaknesses and Missing Experiments

**Single model family.** All experiments use LLaMA or Llama 2 architectures. The paper claims in Section 1 that YaRN is applicable to "the LLaMA, the GPT-NeoX, and the PaLM families of models," but provides zero evidence for GPT-NeoX or PaLM. The wavelength thresholds ($\alpha = 1$, $\beta = 32$) and temperature formula ($\sqrt{1/t} = 0.1 \ln(s) + 1$) were tuned on LLaMA-family models and may not transfer. This is a significant gap between claimed generality and demonstrated generality.

**No sensitivity analysis for key hyperparameters.** The paper does not explore how performance varies with $\alpha$ and $\beta$ (the NTK-by-parts thresholds), with different temperature scaling formulas, or with different training data context lengths for the same target extension factor. The robustness of the method to these choices is unknown. If the method is highly sensitive to $\beta$ (say, performance collapses at $\beta = 16$ or $\beta = 64$), then adopting YaRN for a new architecture would require costly hyperparameter sweeps — exactly the problem the paper criticizes NTK-aware for.

**No perplexity evaluation beyond 128k.** The $s = 32$ models are evaluated only to 128k (their nominal target). Whether perplexity would remain stable at 256k or 512k is unknown. Dynamic Scaling is designed to handle this gracefully, but no evidence is provided. Given the paper's emphasis on extrapolation (training on 64k, testing on 128k), testing the limits of that extrapolation would be natural and informative.

**Small sample sizes for key metrics.** GovReport evaluations use 50 documents (Table 8) — reasonable. Proof-pile perplexity uses 10 documents (Tables 1, 5, 7) — small but perhaps sufficient given the 128k-token length of each document. Passkey retrieval uses 10 iterations for the largest models (Table 9) — too few for precise accuracy estimates. The paper does not report confidence intervals for any metric, making it impossible to assess whether the 0.49% score drop between $s = 16$ and $s = 32$ (Section 4.4) is statistically meaningful or noise.

**Confounded benchmark comparison with Code Llama.** Table 10 compares YaRN models against Code Llama on natural language benchmarks. Code Llama was fine-tuned on code, which explains its catastrophic drops on HellaSwag (60.8 vs. 77.8 for Llama 2 7B base) and ARC-c (39.9 vs. 53.1). The paper includes these numbers but does not caveat them adequately — the comparison is between a context extension method applied to a general-purpose model (YaRN on Llama 2) and a context extension method applied to a code-specialized model (NTK-aware on Code Llama). The performance differences are dominated by base model capabilities, not extension method quality.

**No ablation isolating the PG19 fine-tuning effect.** All fine-tuned models are trained on PG19, which is stylistically very different from the LLaMA/Llama 2 pretraining data. The MMLU drops (Table 2: 35.7 → 30.0 for YaRN, 35.7 → 25.9 for PI) cannot be attributed solely to the interpolation method — they partially reflect catastrophic forgetting from domain shift. A control experiment fine-tuning the base model on PG19 with *no* position encoding change (i.e., $s = 1$, 2k-context PG19 chunks) would separate the forgetting effect from the interpolation effect. Without this, the "preservation" claim is confounded.

**No evaluation of computational overhead for Dynamic Scaling.** While the paper correctly notes that Dynamic Scaling with kv-caching requires caching pre-RoPE embeddings (Section 3.4), it does not measure the memory or latency implications of this choice. Standard implementations cache post-RoPE key/value vectors; switching to pre-RoPE caching may require code changes in inference frameworks and could affect memory bandwidth. The claim of "zero overhead" applies to the fixed-$s$ case but may not fully apply to Dynamic Scaling.

**The 400-step training budget is somewhat arbitrary.** The paper demonstrates that 400 steps is sufficient for YaRN, but does not show whether 200 steps would be sufficient, or whether 800 steps would yield further improvements. The training curves in Figure 2 show that loss is still decreasing at step 400 for all methods, suggesting that additional training might improve performance further. The efficiency comparison against PI at 1,000 steps is valid, but the paper does not establish that 400 steps is the *optimal* training budget for YaRN — only that it is sufficient to surpass PI at 1,000 steps.

## 6. Limitations and Trade-offs

### Single Model Family: No Evidence Beyond LLaMA/Llama 2

**The assumption or constraint.** The paper states in Section 1 that YaRN is applicable to "the LLaMA, the GPT-NeoX, and the PaLM families of models," but all experiments — every perplexity evaluation, every passkey retrieval test, every benchmark comparison — use only LLaMA 7B (Touvron et al., 2023a) and Llama 2 7B/13B (Touvron et al., 2023b). GPT-NeoX and PaLM are never trained or evaluated. The wavelength thresholds ($\alpha = 1$, $\beta = 32$) and the temperature scaling formula ($\sqrt{1/t} = 0.1 \ln(s) + 1$) were determined empirically on LLaMA-family models, and the paper acknowledges this scope only implicitly — the formula "was found by fitting... on LLaMA 7b, 13b, 33b and 65b models" (Section 3.3) and the thresholds were "found experimentally that for the Llama family of models, good values for $\alpha$ and $\beta$ are $\alpha = 1$ and $\beta = 32$" (Section 3.2).

**The consequence.** The wavelength ratio $r(d) = L / \lambda_d$ depends on three architectural parameters: the pretrained context length $L$, the hidden dimension size $|D|$, and the RoPE base $b = 10000$. For GPT-NeoX or PaLM, which may use different $|D|$ and $L$ values, the distribution of $r(d)$ across dimensions shifts. The thresholds $\alpha = 1$ and $\beta = 32$ — chosen because $\alpha = 1$ corresponds to the dimension whose wavelength equals $L$, and $\beta = 32$ was found to work for Llama — may not generalize. A model with a much larger $|D|$ would have more dimensions with very long wavelengths ($r(d) \ll 1$), potentially requiring different fractionation between the interpolated and preserved regimes. A model with a different $b$ would shift all wavelengths proportionally. The temperature formula was fitted only to LLaMA/Llama 2 perplexity curves and may not be universal — the paper notes it "may have certain degree of 'universality'" but provides no cross-architecture evidence. A practitioner adopting YaRN for a non-LLaMA RoPE-based model cannot rely on the paper's hyperparameters and would need to replicate the fitting procedure (sweeping $\alpha$, $\beta$, and temperature values), which the paper itself criticizes NTK-aware interpolation for requiring: "it is very difficult to determine what optimal base should be used... the best base... usually has to be found empirically, which significantly increases the difficulty and cost" (Section 3.1). YaRN may inherit a milder version of this same problem for new architectures.

**What evidence exists in the paper.** None. The paper provides zero experiments on GPT-NeoX, PaLM, or any non-LLaMA architecture. The claims of broader applicability are stated in the introduction and conclusion but are entirely unsubstantiated.

**Mitigation status.** Not addressed. The paper does not discuss how to adapt $\alpha$, $\beta$, or the temperature formula to other model families, nor does it provide guidance on how sensitive these parameters are to architectural differences. This is a gap between the claimed generality and the demonstrated scope.

---

### The 10× Token Efficiency Claim Is Not Directly Measured

**The assumption or constraint.** The headline claim — "requiring 10× less tokens and 2.5× less training steps than previous methods" (Abstract) — aggregates two separate comparisons that are not equally well-supported. The 2.5× training step reduction is directly measured in Table 6: at $s = 2$ (4k → 8k), YaRN at 400 steps achieves 3.35 perplexity vs. PI at 1,000 steps achieving 3.34 — essentially identical performance with 2.5× fewer steps, a clean and convincing controlled comparison. The "10× fewer tokens" claim, however, is not isolated in any single experiment. It appears to combine the 2.5× step reduction with the shorter training sequences YaRN uses: the $s = 16$ and $s = 32$ models are trained on 64k-token segments, whereas previous methods (the paper implies) required training on the full target context length. For the $s = 32$ model targeting 128k context, training on 64k segments halves the tokens per sequence compared to training on 128k segments. Combined with the 2.5× step reduction, this could yield roughly 5× fewer tokens, but not obviously 10×. The paper does not provide a line-item breakdown of total tokens processed for each method.

**The consequence.** The 10× figure is a headline number that shapes how the paper is received and cited, but a practitioner trying to reproduce the efficiency gains cannot verify or budget for them without knowing exactly what they refer to. Is it 10× fewer tokens than PI at the same extension factor? Than NTK-aware? Than Chen et al. (2023) specifically? The training data context length for PI baselines is not reported — Chen et al. (2023) may have trained on 16k-token segments for their $s = 8$ extension, but this is not stated in the paper. The 10× claim is plausible but unverifiable from the reported data alone. This matters because the paper's primary practical contribution is efficiency — if the efficiency claim is imprecise or not reproducible across settings, the core value proposition is weakened.

**What evidence exists in the paper.** Table 6 provides direct evidence for the 2.5× step reduction at $s = 2$. Table 4 compares total A100 GPU-hours: YaRN to 64k costs 256 hours, while Chen et al. (2023) PI to 16k costs 640 hours — a 2.5× reduction in GPU-hours for a 4× larger extension factor, which is consistent with the efficiency narrative but does not directly measure token counts. The paper does not report total tokens processed for any training run.

**Mitigation status.** Partially addressed. The 2.5× step reduction is well-supported. The 10× token figure is stated without derivation and cannot be verified from the paper's reported data. Future work citing this figure should treat it as an order-of-magnitude estimate rather than a precisely measured quantity.

---

### Training Data Distribution Shift Confounds Benchmark Preservation Claims

**The assumption or constraint.** All fine-tuned models are trained on PG19 (Rae et al., 2020), a dataset of pre-1919 books that the paper itself acknowledges is "very different from the original pre-training dataset used for LLaMA and Llama 2 models" (Section 4.4). The benchmark evaluations (Tables 2, 3, 10) compare fine-tuned YaRN models against the base (non-fine-tuned) models on ARC-Challenge, HellaSwag, MMLU, and TruthfulQA. These benchmarks measure capabilities — reasoning, commonsense knowledge, factual knowledge — that are sensitive to the model's training distribution. Fine-tuning on out-of-distribution text can cause catastrophic forgetting: the model loses capabilities it acquired during pretraining because the fine-tuning gradients overwrite parameters relevant to those capabilities, even if the task loss on PG19 is low.

**The consequence.** The reported drops on standardized benchmarks — particularly MMLU, which shows the largest degradation across all methods and model sizes (Table 2: LLaMA 7B base 35.7 → YaRN 30.0; Table 3: Llama 2 7B base 43.8 → YaRN $s = 16$ 42.5 → YaRN $s = 32$ 41.7) — cannot be attributed solely to the position encoding modification. Some fraction of the degradation is almost certainly due to catastrophic forgetting from domain shift during PG19 fine-tuning, not from the interpolation method. The paper conflates these two effects when it claims that YaRN "preserves" original abilities. A practitioner who needs to extend context while retaining strong MMLU performance cannot determine from the paper's evidence whether switching to an in-domain fine-tuning dataset (e.g., RedPajama, which is a closer match to LLaMA's pretraining data) would recover the lost MMLU points, or whether the position encoding modification itself imposes an irreducible penalty.

**What evidence exists in the paper.** The benchmark tables show that MMLU drops are the largest across all methods — even NTK-by-parts, which the paper presents as a strong method, drops from 35.7 to 32.7 on LLaMA 7B (Table 2). PI drops to 25.9. The fact that the rank order of methods by MMLU preservation (NTK-by-parts ≈ YaRN > NTK-aware > PI) correlates with their overall quality does not isolate the forgetting effect from the interpolation effect. The paper does not report a control experiment: fine-tuning the base model on PG19 with *no position encoding change* ($s = 1$, using 2k-context PG19 chunks for LLaMA or 4k-context chunks for Llama 2) and measuring the resulting benchmark drops. This control would quantify the pure forgetting effect, and the residual drop beyond that could be attributed to the interpolation method. Without it, the claim "YaRN preserves original abilities" is confounded.

**Mitigation status.** Not addressed. The paper acknowledges the PG19 distribution difference but treats it as a feature (demonstrating robustness to domain shift) rather than a confound for benchmark evaluation. The claim of minimal degradation is qualified in Section 4.4 ("Some variance is to be expected as the PG19 dataset... is very different") but the magnitude of the forgetting-vs.-interpolation effect is not estimated.

---

### No Characterization of the Extrapolation Limit Beyond the Training Context Length

**The assumption or constraint.** The paper's most surprising result is that YaRN $s = 32$ models trained on 64k-token sequences successfully process 128k-token sequences at inference (Table 1, Table 9). This is presented as a strength — "train short, test long" — and it genuinely is. However, the paper does not characterize *how far* this extrapolation extends. The $s = 32$ model is evaluated up to 128k (its nominal target) but not beyond. The $s = 16$ model (trained on 64k, targeting 64k) is evaluated at 128k in Table 1 and produces >100 perplexity, confirming that a model fine-tuned for a specific extension factor fails completely at longer lengths. But for the $s = 32$ model, we do not know: does perplexity remain stable at 256k? Does passkey retrieval remain accurate? Is there a second cliff beyond 128k, or does Dynamic Scaling combined with $s = 32$ training provide graceful degradation indefinitely?

**The consequence.** A practitioner deploying a YaRN $s = 32$ model cannot know what happens if a user provides a 200k-token document. The model might process it correctly (if the extrapolation continues), might experience a sharp perplexity cliff at some point beyond 128k (if the $s = 32$ interpolation has an effective maximum), or might degrade gradually (if Dynamic Scaling handles it). The paper's philosophy — that models should gracefully handle variable-length inputs via Dynamic Scaling — suggests the third outcome, but no evidence supports it. This is a practical deployment concern: production systems need to know whether to truncate inputs at 128k, whether to set a hard limit or a soft threshold, and what error modes to expect.

**What evidence exists in the paper.** None beyond 128k. Table 1 stops at 131,072 tokens. Figure 7 shows perplexity up to the nominal context window of each model. Figure 8 (Dynamic Scaling on non-fine-tuned Llama 2) shows stable perplexity to approximately 16k from a 4k pretrained model, but this is only a 4× extension, and it uses a single document. The paper does not push any model to failure to characterize the extrapolation boundary.

**Mitigation status.** Not addressed. The paper emphasizes the "train short, test long" result as a positive finding but does not explore its limits. The Dynamic Scaling technique is explicitly designed for graceful degradation beyond the fine-tuned length, but this property is never tested for the fine-tuned models at their largest extension factors.

---

### The Wavelength Thresholds ($\alpha$, $\beta$) Are Not Systematically Validated

**The assumption or constraint.** The NTK-by-parts interpolation — which YaRN inherits — partitions RoPE dimensions into three regimes based on the ratio $r(d) = L / \lambda_d$ using two thresholds: $\alpha = 1$ and $\beta = 32$ (Section 3.2). Dimensions with $r(d) < 1$ (wavelength longer than the pretrained context) are fully interpolated; dimensions with $r(d) > 32$ are not interpolated at all; dimensions in between receive a linear blend. The choice of $\alpha = 1$ has a principled justification — it is the point where wavelength equals context length, the boundary between dimensions that complete less than one full rotation during training (absolute position encoders) and those that complete more (relative position encoders). The choice of $\beta = 32$ has no such justification — the paper states it was "found experimentally that for the Llama family of models, good values for $\alpha$ and $\beta$ are $\alpha = 1$ and $\beta = 32$" but provides no data showing the effect of different $\beta$ values on perplexity or benchmark performance.

**The consequence.** The sensitivity of YaRN's performance to $\beta$ is unknown. If $\beta = 16$ were used instead, would performance degrade substantially? If $\beta = 64$, would it improve? The $\beta$ parameter controls how many dimensions are partially interpolated vs. fully preserved. Setting $\beta$ too low would interpolate dimensions that primarily encode relative position, damaging local attention resolution (the same failure mode as PI). Setting $\beta$ too high would fail to interpolate dimensions that still have a meaningful absolute-position component, causing out-of-distribution position encodings at extended lengths. The paper's criticism of NTK-aware interpolation — that "it is very difficult to determine what optimal base should be used" — applies in a milder form to its own method: a practitioner adapting YaRN to a new architecture with a different $|D|$ or $b$ cannot simply use $\alpha = 1$, $\beta = 32$ and expect optimal results without validation, yet the paper provides no guidance on how to tune these parameters or how sensitive the method is to them.

**What evidence exists in the paper.** None. No sweep over $\alpha$ or $\beta$ values is reported. The experimental section contains no sensitivity analysis for these thresholds. The statement that $\alpha = 1$ and $\beta = 32$ are "good values" is asserted without supporting data.

**Mitigation status.** Not addressed. The paper treats these thresholds as fixed constants for the LLaMA family and does not discuss their sensitivity, tuning methodology, or expected values for other architectures. This is a gap in the method's reproducibility for non-LLaMA models.

---

### Passkey Retrieval Sample Sizes Are Too Small for Reliable Accuracy Estimates at Large Scales

**The assumption or constraint.** The passkey retrieval task directly measures whether the model can attend to and retrieve information from arbitrary positions within the extended context — arguably the most practically relevant capability for long-context models. For the 32k LLaMA 7B ablation experiments, the paper uses 50 iterations per context length (Figure 3, right), which yields reasonably stable accuracy estimates: a 95% confidence interval for 90% accuracy based on 50 trials would be approximately ±8 percentage points. For the 64k and 128k Llama 2 models — the headline results — the paper uses only 10 iterations per context length (Table 9). At 10 trials, a single failure drops accuracy by 10 percentage points. The reported 99.4% accuracy for YaRN $s = 32$ at 128k context means the model succeeded on 10 out of 10 trials — a strong signal, but one where the lower bound of a reasonable confidence interval could be as low as ~70%, depending on the assumed distribution.

**The consequence.** The passkey retrieval results for the largest models — the ones that demonstrate the paper's most impressive extension factors — are imprecise. The difference between 96.3% (YaRN $s = 16$) and 99.4% (YaRN $s = 32$) is based on roughly one additional failure in the $s = 16$ condition across 10 trials — well within sampling noise. The paper's interpretation that the $s = 16$ model "might be relatively undertrained for the passkey retrieval task" (Appendix B.5) may be correct, but it is also consistent with random variation. A practitioner deciding which YaRN variant to deploy cannot confidently distinguish a 96% retrieval model from a 99% retrieval model based on 10 trials. For Code Llama's reported 94.3% accuracy at 112k context, the confidence interval is even wider, and the comparison with YaRN's 99.4% — which the paper presents as evidence of YaRN's superiority — is statistically fragile.

**What evidence exists in the paper.** Table 9 reports passkey accuracies based on 10 iterations per context length for the 64k and 128k models. Figure 3 (right) uses 50 iterations for the 32k LLaMA 7B models, showing much smoother curves. The paper does not report confidence intervals or discuss the statistical reliability of the passkey results.

**Mitigation status.** Not addressed. The reduction from 50 to 10 trials was likely driven by compute constraints — evaluating a 128k-context model with many passkey positions is expensive — but the paper does not acknowledge the resulting precision loss. The passkey results should be interpreted as qualitative indicators ("the model can retrieve the passkey at this length") rather than precise accuracy measurements.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

YaRN shifts the conversation around transformer context extension from "how much training do we need?" to "what does each RoPE dimension actually encode?" This is a **conceptual reframing**, not a paradigm shift — the paradigm (interpolate position encodings, fine-tune briefly) remains the same one established by Chen et al. (2023). What changes is the diagnostic precision brought to bear on that paradigm: before YaRN, RoPE dimensions were treated as an undifferentiated block to be stretched or compressed uniformly; after YaRN, they are recognized as a heterogeneous population where some dimensions encode absolute position, others encode relative position, and the boundary between these regimes is predictable from the wavelength ratio $r(d) = L / \lambda_d$.

The field-level impact of this reframing is threefold:

**First, it resolves the tension between PI and NTK-aware interpolation.** PI's uniform stretching destroys high-frequency relative-position information, capping practical extension at roughly $s = 8$. NTK-aware's base change preserves high frequencies but extrapolates some dimensions, making fine-tuning unstable. The paper's wavelength analysis explains *why* both methods have ceilings — PI damages the wrong dimensions, NTK-aware fixes this imprecisely — and NTK-by-parts/YaRN addresses the root cause by treating dimensions according to their functional role. This converts a confusing empirical tradeoff (PI works better with fine-tuning, NTK-aware works better without) into a coherent diagnostic framework. The prior literature's contradictory recommendations — fine-tune with PI, deploy without fine-tuning with NTK-aware — are reconciled by recognizing that neither method correctly handles all dimensions; each is merely lucky in different regimes.

**Second, it establishes that attention calibration is a distinct failure mode from position encoding.** The temperature scaling component of YaRN — $\sqrt{1/t} = 0.1 \ln(s) + 1$ — addresses a problem that no prior method even identified: even with correct position mappings, the softmax distribution's peakedness becomes miscalibrated at extended lengths. This is conceptually significant because it separates context extension into two sub-problems (position encoding and attention calibration) that can be addressed independently and combined. The fact that the temperature formula transfers across model sizes (7B through 65B) and families (LLaMA and Llama 2) without modification suggests it captures something fundamental about how attention uncertainty scales with position encoding stretch — a finding that invites theoretical analysis rather than just empirical curve-fitting. It also implies that future context extension methods, whether based on RoPE or other position encodings, should probably include an attention-calibration component.

**Third, it demonstrates that training context length and inference context length can be decoupled.** The $s = 32$ model trained on 64k-token segments and evaluated successfully at 128k (Table 1, Table 9) is the paper's most striking single result, and it changes the economics of long-context model development. The standard assumption — that you must train at the target context length — implied that doubling the context window quadruples the training attention cost (due to the $O(L^2)$ scaling of self-attention). YaRN's result suggests that the model learns the *interpolation scheme*, not a specific sequence length, and that training data context length can be chosen based on memory constraints rather than target deployment length. This makes 128k, 256k, or potentially longer context windows achievable with training costs closer to 64k than to the target length — a practical breakthrough for resource-constrained research groups and a conceptual one for understanding what the model actually learns during context extension fine-tuning.

The paper also makes several research directions **less attractive**:

- **Complex attention mechanism modifications** (ReRoPE, LM-Infinite) lose appeal when a pure embedding-level method achieves comparable or better results while maintaining Flash Attention 2 compatibility. The paper does not explicitly argue this, but the practical community's strong preference for Flash Attention compatibility — evidenced by the rapid adoption of YaRN in open-source projects — suggests that methods requiring custom attention kernels face an uphill adoption battle.
- **Brute-force long-context pretraining** (Xiong et al., 2023, at 64,000 A100-hours) looks increasingly difficult to justify when YaRN achieves 128k context in 384 A100-hours. The efficiency ratio (roughly 150×) is so large that even substantial improvements to the pretraining approach would struggle to close the gap for extension factors within YaRN's demonstrated range.
- **Empirical base-frequency tuning** (the NTK-aware approach adopted by Code Llama) is superseded by the explicit wavelength-threshold method, which eliminates the need for costly hyperparameter sweeps — though the paper does not fully escape this criticism, since $\beta = 32$ was itself found empirically for LLaMA models.

### Follow-Up Research This Work Enables

**Characterize the extrapolation ceiling for "train short, test long."** The $s = 32$ model trains on 64k and generalizes to 128k, but the paper never tests whether this extrapolation continues to 256k, 512k, or beyond. A direct follow-up would take the existing YaRN $s = 32$ checkpoints (which the paper makes available) and evaluate sliding-window perplexity and passkey retrieval at 192k, 256k, 384k, and 512k context lengths on Proof-pile documents that are long enough to support those windows. The key question: is there a second perplexity cliff beyond 128k, or does Dynamic-YaRN's scaling ($s = \max(1, l'/L)$ per forward pass) provide graceful degradation indefinitely? A negative result — a sharp cliff at, say, 200k — would establish that the model's generalization is bounded by the effective range of the $s = 32$ interpolation, not just the training data length. A positive result — stable perplexity to 512k — would fundamentally change the conversation about what context lengths are achievable with modest fine-tuning.

**Isolate the catastrophic forgetting effect from the interpolation effect on benchmark performance.** The paper's benchmark evaluations (Tables 2, 3, 10) show that MMLU drops by 2–6 points across all extension methods, with the paper attributing this to the PG19 fine-tuning distribution shift. The control experiment is straightforward and was not done: fine-tune the base Llama 2 7B model on PG19 for 400 steps with the original RoPE frequencies and no context extension (2k-context PG19 chunks for LLaMA, 4k for Llama 2), then evaluate on the same standardized benchmarks. The difference between this control model and the base model quantifies pure catastrophic forgetting from PG19 domain shift. Subtract that from the YaRN benchmark drops, and the residual measures the true cost of the position encoding modification itself. This cleanly separates the two confounded effects and would tell practitioners whether switching to an in-domain fine-tuning dataset (e.g., RedPajama, which is a closer match to LLaMA's pretraining data) could recover the lost benchmark points, or whether the interpolation imposes an irreducible penalty. A strong follow-up would run this control for both LLaMA 7B and Llama 2 7B, and also test whether fine-tuning on RedPajama instead of PG19 preserves MMLU while still achieving the same context extension.

**Extend YaRN to non-LLaMA RoPE architectures with systematic threshold validation.** The paper claims applicability to GPT-NeoX and PaLM but provides zero evidence. A direct replication on GPT-NeoX-20B (Black et al., 2022) — which uses RoPE with $b = 10000$ like LLaMA but has $|D| = 6144$ (vs. LLaMA 7B's $|D| = 4096$) and a larger pretrained context length — would test whether the wavelength thresholds $\alpha = 1$ and $\beta = 32$ transfer or need adjustment. The key measurement: sweep $\beta$ over $\{2, 4, 8, 16, 32, 64, 128\}$ while keeping $\alpha = 1$, fine-tune at $s = 8$ for 400 steps on PG19, and evaluate perplexity at 4×, 8×, and 16× the pretrained context. The resulting curve would show how sensitive performance is to $\beta$, whether the optimal value depends on architecture, and whether there's a simple rule (e.g., "set $\beta$ such that the $(\beta)$-threshold dimension has wavelength $L / \beta$ tokens") that generalizes across model families. A negative result — performance is highly sensitive to $\beta$ and the optimal value varies unpredictably across architectures — would significantly weaken the paper's generality claims and suggest that YaRN requires per-architecture tuning, undermining its "drop-in replacement" positioning.

**Develop a theoretically-grounded attention temperature scaling.** The formula $\sqrt{1/t} = 0.1 \ln(s) + 1$ is empirical — fitted to LLaMA/Llama 2 perplexity curves — and the paper provides no theoretical justification for why attention miscalibration occurs or why a logarithmic correction is appropriate. A theoretical follow-up would analyze how the distribution of query-key dot products changes as RoPE frequencies are stretched, and whether the softmax temperature needs to scale to maintain a target entropy or attention-sparsity level. Concretely: take a base LLaMA model, feed it sequences at various lengths $l'$ with YaRN interpolation at fixed $s = L'/L$, and measure the empirical entropy of the attention distribution (averaged across heads and layers) as a function of $l'$ and $s$. If the entropy systematically decreases (attention becomes overconfident) as $l'$ grows beyond $L$, that would mechanistically explain why increasing temperature helps — it counteracts an overconfidence bias induced by out-of-distribution position encodings. If the entropy changes in a different pattern, the temperature scaling might be compensating for a different phenomenon entirely. A strong theoretical result would derive the optimal temperature from first principles (e.g., maintaining constant per-head attention entropy across context lengths) and test whether the derived formula matches the empirical $\sqrt{1/t} = 0.1 \ln(s) + 1$. If it does, the temperature scaling generalizes beyond LLaMA; if not, the formula may be architecture-specific and new architectures would need their own fitting.

**Combine YaRN with retrieval-based or hierarchical attention for even longer contexts.** YaRN achieves 128k context, but many real-world applications — multi-document legal analysis, full-codebase understanding, scientific literature review — require 500k+ tokens. A natural extension is to use YaRN for the local attention window (e.g., 128k tokens) within a hierarchical or retrieval-augmented architecture that handles longer documents by chunking and cross-chunk attention. The concrete experiment: take the YaRN $s = 32$ Llama 2 7B model, use it as the base encoder in a retrieval-augmented generation pipeline where long documents are split into 128k-token chunks with overlap, each chunk is encoded with YaRN, and cross-chunk attention is handled by a lightweight retriever or a dedicated cross-attention module trained on top of the frozen YaRN encoder. Evaluate on a benchmark requiring very long context (e.g., NarrativeQA, SummScreenFD, or a constructed passkey retrieval task at 500k tokens) and compare against: (a) the same architecture with a shorter-context base model, (b) a from-scratch long-context model at equivalent total parameters. This would test whether YaRN's context extension composes with architectural scaling to reach lengths far beyond what either approach achieves alone.

**Stress-test YaRN with adversarial or structured long-context tasks that probe attention quality.** Perplexity and passkey retrieval measure different things: perplexity can be low even if the model ignores distant context (as the NTK-aware results in Table 5 and Figure 3 demonstrate), and passkey retrieval only tests whether a single piece of information can be retrieved, not whether the model can integrate information from multiple distant locations. A stronger evaluation would use tasks that require synthesizing information from two or more widely separated positions — for example, the "two-passkey" variant where the model must retrieve and add two numbers placed at different random locations, or a multi-hop question-answering task where evidence is distributed across a 128k-token document. The key measurement: pass@k accuracy as a function of distance between the relevant pieces of information, comparing YaRN-extended models against Code Llama (NTK-aware, ~100k) and Together.ai (PI, 32k). A negative result — YaRN models fail at multi-hop retrieval even when single-passkey accuracy is high — would indicate that the method achieves long-range *attention* but not long-range *reasoning*, which is a practically important distinction for downstream applications. Conversely, strong multi-hop performance would substantially strengthen the case that YaRN models genuinely "understand" long contexts rather than merely not ignoring them.

### Practical Applications and Downstream Use Cases

**Cost-efficient fine-tuning of long-context open-source models for domain-specific applications.** The paper's training efficiency numbers (Table 4: 256 A100-hours for Llama 2 7B to 64k, 384 A100-hours to 128k) mean that a small team with access to a single 8×A100 node can extend a 7B model to 128k context in under 2 days. The 10× token reduction compared to PI means the training dataset itself can be smaller and cheaper to curate. This enables a deployment pattern where an organization takes an open-source base model (Llama 2, or its successors), extends the context window using YaRN on their own domain-specific long documents (legal contracts, medical records, technical documentation), and then optionally fine-tunes further on task-specific data — all within a feasible compute budget. The fact that YaRN preserves short-context benchmark performance (Table 3: ARC-c drops only 0.8 points for Llama 2 7B at $s = 16$) means the extended model does not sacrifice general capabilities for long-context specialization. A legal-tech company could take Llama 2 7B, extend to 128k with YaRN, fine-tune on legal documents, and deploy a model that handles full contracts end-to-end for roughly $50–100 in cloud compute costs — a 100× cost reduction compared to the $6,400–64,000 A100-hours reported for NTK-aware pretraining-scale approaches (Table 4).

**On-the-fly context extension for variable-length workloads without model modification.** Dynamic-YaRN (Section 3.4, Figure 8) enables a base Llama 2 model — with no fine-tuning whatsoever — to process sequences 2–4× longer than its 4k pretrained context window with stable perplexity. This matters for deployment scenarios where: (a) the model cannot be modified (e.g., using a third-party API or a frozen deployment), (b) occasional long inputs need to be handled but most are short, or (c) the exact maximum context length is unpredictable. A customer support chatbot built on a Llama 2 base could use Dynamic-YaRN to handle users who paste lengthy error logs or conversation histories, automatically scaling position encodings to the current sequence length without requiring a separate long-context model variant. The implementation cost is minimal — the paper provides code, and the change is confined to the rotary embedding generation, with zero inference overhead when using fixed-$s$ mode — making this a "free" upgrade for any RoPE-based deployment. The paper's Figure 8 shows Dynamic-YaRN maintaining perplexity below 10 at 16k tokens from a 4k-trained model, which is usable for retrieval-style tasks where exact token prediction quality is less critical than maintaining coherent attention patterns.

**Iterative self-improvement and data generation pipelines that process long documents.** Many self-improvement methods (STaR, ReST, rejection sampling fine-tuning) require generating high-quality completions on training data. When the training data includes long documents — books, code repositories, research papers — a short-context model must chunk them, losing cross-chunk coherence. YaRN enables a single model to process entire documents up to 128k tokens in one pass, generating labels, summaries, or rationales that are globally consistent rather than chunked. The $s = 32$ model's ability to train on 64k and generalize to 128k (Table 1) is particularly valuable here: the data generation budget can focus on 64k-token segments (cheaper to process), and the generated labels can be validated at 128k during a separate evaluation pass. For a research lab building a paper-summarization dataset, this means they can fine-tune a YaRN-extended model once (384 A100-hours), use it to generate summaries of full papers (typically 10k–30k tokens, well within the 128k window), and then use those summaries to train a smaller, faster student model — all without the train-generate feedback loop being bottlenecked by context length. The passkey retrieval results (99.4% accuracy at 128k, Table 9) provide confidence that the model genuinely attends to the full document rather than generating summaries from the first few pages.
