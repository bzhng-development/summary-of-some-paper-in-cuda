# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**ArXiv:** [2402.13753](https://arxiv.org/abs/2402.13753)

## 🎯 Pitch

LongRoPE presents a breakthrough method to expand the usable context window of large language models from a mere few thousand to over two million tokens—more than 500× previous limits—without requiring massive retraining at ultra-long lengths. By intelligently learning dimension- and position-specific rescalings of rotary position embeddings, coupled with a progressive fine-tuning and interpolation strategy, LongRoPE preserves short-context performance while unlocking unprecedented long-range reasoning. This leap enables LLMs to handle tasks like book-length analysis, multi-document reasoning, and long-form conversations, advancing the capabilities of existing models for real-world, memory-intensive applications.

---

## 1. Executive Summary

This paper introduces **LongRoPE**, a method that extends the context window of pre-trained LLMs to an unprecedented 2048k tokens — a 512× increase over the standard 4k limit — while requiring only up to 1k fine-tuning steps at training lengths within 256k. The approach operates on LLaMA2-7B and Mistral-7B and is built on three named innovations: **non-uniform positional interpolation** exploiting two forms of non-uniformities — varying RoPE dimensions and initial token positions — discovered through an efficient evolutionary search (e.g., searching for per-dimension rescale factors λᵢ and a token-position threshold ˆn to preserve crucial high-frequency dimensions and attention-sink tokens), a **progressive extension strategy** that first fine-tunes a 256k model then applies a second non-uniform interpolation to reach 2048k without further fine-tuning (e.g., leveraging the method's 8× non-fine-tuning extension capability on the already-extended checkpoint), and a **shorter context window recovery** procedure that readjusts RoPE rescale factors for 8k-and-below lengths to restore original short-context performance. LongRoPE achieves over 90% passkey retrieval accuracy at 2048k tokens and maintains perplexity comparable to or better than state-of-the-art baselines at all lengths from 4k to 256k, establishing that extremely long context extension is feasible without training on correspondingly long texts — the progressive strategy exploiting the 8× non-fine-tuning property proves sufficient.

## 2. Context and Motivation

### The Core Problem: How Do You Extend an LLM's Context Window Past 128k Tokens?

The paper addresses a specific technical bottleneck: current methods for extending the context window of pre-trained LLMs hit a hard ceiling around 128k tokens. This limitation is not incidental—it stems from three intertwined obstacles that compound as the target extension ratio grows.

**Obstacle 1: Catastrophic values from untrained position indices.** When an LLM like LLaMA2 (trained with a 4k context window) is asked to process a 2048k-token document, over 99.8% of the position indices it encounters were never seen during training. The RoPE positional embedding assigns these novel positions rotation angles that the model's attention mechanisms have no experience processing, introducing out-of-distribution values that cause perplexity to spike and fine-tuning to become unstable. The paper notes this challenge is particularly acute at extreme extension ratios: going from 4k to >1000k introduces more than 90% new positions.

**Obstacle 2: Scarcity of extremely long training texts.** Fine-tuning for extended context windows typically requires training data at the target length. But texts exceeding 1000k tokens are rare in public datasets, making supervised fine-tuning at these lengths impractical. Even if such texts existed, training on them would be prohibitively expensive—the quadratic complexity of standard attention means that doubling the context length quadruples the computational cost per sample.

**Obstacle 3: Attention dispersion degrades short-context performance.** When a model's context window is radically extended, its attention mechanism must spread representational capacity across vastly more token positions. This "thin spreading" causes performance on short sequences (where the model originally excelled) to degrade—a known phenomenon from prior work that the paper must contend with. At a 512× extension ratio, positions within the original 4k window become extremely "crowded" in the RoPE embedding space, losing the representational resolution the model was originally trained with.

### Why This Matters: Applications Requiring Million-Token Contexts

The paper motivates the importance of extreme context extension by pointing to concrete application classes that fundamentally require processing very long documents:

- **In-context learning with numerous examples.** As the number of exemplars grows, the prompt length expands linearly. Tasks requiring hundreds of examples (or examples that are themselves long documents) quickly exhaust even 128k windows.
- **LLM agents operating over persistent memory.** Agent architectures that maintain interaction histories, environment states, and tool outputs accumulate tokens rapidly—a single complex agent session with external API calls can easily exceed 100k tokens.
- **Long document understanding.** Summarization, question answering, and analysis over book-length texts, legal documents, or codebases with millions of tokens are simply impossible with shorter context windows, forcing workarounds like chunking and retrieval that sacrifice global coherence.

The 128k barrier means these applications either require architectural modifications that break compatibility with existing LLM infrastructure or force users into retrieval-augmented approaches that lose cross-document reasoning capabilities.

Beyond practical applications, the paper addresses a theoretical concern: whether positional interpolation methods *can in principle* scale to 512× extension ratios, or whether fundamental information-theoretic limits in the RoPE embedding prevent such extreme compression of position information.

### Prior Approaches and Their Limitations

The paper situates itself within the **positional interpolation** lineage, which modifies the RoPE embedding to fit extended position indices into the pre-trained frequency range, rather than requiring the model to learn embeddings for novel positions from scratch. Understanding why each prior method stalls at ~128k reveals the paper's intellectual motivation.

#### Linear Positional Interpolation (PI) — Chen et al., 2023a

PI takes the simplest approach: for a target extension ratio $s$, multiply all rotation angles by $\lambda = s$. This linearly compresses all position indices into the pre-trained range. If the original context was 4k and you want 8k, every position $n$ is treated as position $n/2$.

**Why it fails at scale:** The compression makes position information extremely "crowded"—tokens that are physically far apart in the sequence appear adjacent in the positional embedding. The model loses the ability to distinguish nearby positions because their RoPE angles differ by only tiny amounts (divided by $s$). At $s=128$ (a 512k window), the angular resolution becomes so poor that the model can barely distinguish position $n$ from $n+100$. PI requires fine-tuning to adapt to this crowding, and even with fine-tuning, performance degrades sharply as $s$ grows. The paper notes PI "tends to underperform at high extension ratios."

#### NTK-Aware Interpolation (LocalLLaMA, 2023b;a)

NTK-based methods observe that not all RoPE dimensions are equally important for distinguishing positions. Lower dimensions (higher frequencies) encode fine-grained position information; higher dimensions (lower frequencies) encode coarse position information. Applying the same compression factor $\lambda = s$ to all dimensions wastes representational capacity—the high-frequency dimensions get unnecessarily compressed, while the low-frequency dimensions could handle more compression than PI applies.

The NTK approach applies *unequal* scaling: lower dimensions (high frequency) receive less interpolation (closer to direct extrapolation), while higher dimensions (low frequency) receive more interpolation. The scaling follows $\lambda_i = s^{i/(d/2)}$, where $i$ is the dimension index and $d$ the embedding dimension. This produces a gradual transition from extrapolation at low $i$ to interpolation at high $i$.

**Why it fails at scale:** While NTK improves over PI, it can typically achieve only about a 4× extension in non-fine-tuning settings before perplexity spikes. The formula-based scaling is a fixed one-size-fits-all curve that doesn't adapt to the specific model's learned positional sensitivities. Moreover, it only addresses *dimension-wise* non-uniformity—it doesn't consider that different *token positions* might need different treatment.

**Dynamic NTK** improves on this by adjusting $\lambda$ based on the actual sequence length encountered during inference (LocalLLaMA, 2023a), but still relies on the same fixed parametric scaling curve.

#### YaRN (Peng et al., 2023)

YaRN provides the most sophisticated prior approach by dividing RoPE dimensions into three frequency-based groups, each receiving a qualitatively different treatment:

- **High frequency dimensions** undergo extrapolation ($\lambda = 1$) — no interpolation at all. The rationale is that these dimensions encode the finest position discrimination and cannot afford any compression.
- **Low frequency dimensions** undergo linear interpolation ($\lambda = s$) — full compression, since these dimensions encode coarse position information with redundancy.
- **Middle frequency dimensions** undergo NTK-aware interpolation — the gradual transition between the two extremes.

YaRN also introduces a "temperature" adjustment to the attention softmax to compensate for entropy changes introduced by the new embedding distribution.

**Why it fails at scale:** YaRN represents a significant improvement and is the strongest baseline in the paper's experiments, but two limitations prevent extreme scaling:

1. **Human-designed grouping thresholds are suboptimal.** The boundary between "high," "middle," and "low" frequency groups is determined by empirical heuristics rather than optimized for the specific model. A dimension that happens to be crucial for the model's learned attention patterns might get inappropriate treatment because it falls just on the wrong side of a human-chosen cutoff.

2. **Token-position non-uniformity is ignored entirely.** YaRN treats every token position identically—position 1 and position 100,000 receive the same interpolation factors. But the paper's Finding 2 demonstrates that initial tokens (which serve as "attention sinks" and receive disproportionately large attention scores) benefit from less interpolation, and the optimal number of preserved initial tokens depends on the target length.

The paper's Table 1 and Table 3 provide concrete evidence of these gaps. On PG19 with an 8k extension, YaRN achieves perplexities of 32.64 and 87.89 (Table 1)—worse than both PI and NTK—while a simple search over per-dimension rescale factors drops perplexity to 9.37 and 11.34. This demonstrates that the human-designed groupings in YaRN are actively harmful at certain extension ratios because they either over-interpolate or under-interpolate critical dimensions.

#### Non-Interpolation Approaches

The paper also acknowledges two alternative families of context extension that it considers complementary rather than competitive:

**Retrieval-based approaches** (Tworkowski et al., 2023; Wang et al., 2023; Borgeaud et al., 2022) maintain an external memory module and fetch relevant chunks at inference time. These "need explicit modifications on the LLM architectures" and are limited to tasks where relevance can be determined by retrieval—they cannot support operations requiring holistic attention over the entire document.

**Attention-manipulation approaches** (Han et al., 2023; Xiao et al., 2023) use novel attention masks to mitigate the "attention explosion" caused by novel positions, enabling some degree of extrapolation without position interpolation. The paper cites Streaming LLM and LM-Infinite as examples, but notes these are constrained to the model's original context window and "complementary" to interpolation methods.

### How LongRoPE Positions Itself

The paper makes three specific claims about where prior work falls short, and frames LongRoPE as addressing each gap directly:

**Claim 1: Prior non-uniformity is insufficient and human-designed.** Both NTK and YaRN recognize that different RoPE dimensions need different interpolation treatments, but their non-uniformity is either formulaic (NTK's power-law scaling) or coarsely grouped (YaRN's three frequency bands). The paper argues this "subtle non-uniformity is not effectively leveraged," leading to "information loss and hence limiting the context window size." LongRoPE replaces human heuristics with evolutionary search that discovers per-dimension rescale factors optimized for the specific model and target extension ratio. The search space allows each of the 64 RoPE dimensions (for a 128-dimensional embedding, d/2 = 64 frequency pairs) to independently choose a rescale factor, producing a fine-grained, non-monotonic pattern that no simple formula could capture.

**Claim 2: Token-position non-uniformity has been completely overlooked.** No prior interpolation method treats the first $\hat{n}$ tokens differently from the rest of the sequence. The paper's Finding 2—that preserving the original RoPE for initial tokens improves performance and that the optimal $\hat{n}$ varies with extension ratio—is presented as a novel observation. LongRoPE incorporates $\hat{n}$ as a searched parameter alongside the per-dimension rescale factors, effectively creating a two-dimensional non-uniformity: dimensions vary in *how much* they interpolate, and positions vary in *whether* they interpolate at all, with the threshold $\hat{n}$ marking the boundary.

**Claim 3: The 8× non-fine-tuning property enables a fundamentally different extension strategy.** Prior methods assume that extensions beyond ~4× require fine-tuning at the target length. The paper's Finding 3—that optimized non-uniform interpolation can achieve an 8× extension without fine-tuning—is leveraged as more than just a convenience: it is the enabling mechanism for the progressive extension strategy. By fine-tuning a model at 256k (a 64× extension, which requires training), and then applying the 8× non-fine-tuning extension on top of that, LongRoPE reaches 2048k (64× × 8× = 512×) without ever training on data longer than 256k tokens. This elegantly sidesteps the scarcity of >1M-token training texts and the prohibitive cost of training at those lengths.

The paper's language in the introduction positions this explicitly:

> "long texts in current datasets, especially those exceeding 1000k, are limited. Moreover, training on extra-long texts is computationally expensive, requiring prohibitively extensive training hours and GPU resources."

LongRoPE's progressive strategy is the direct response: don't try to train at 2048k; instead, train at 256k (which is achievable with existing data and hardware) and then use non-uniform interpolation to cover the remaining gap. This is only possible because the method's per-dimension, per-position optimization preserves enough RoPE information that an 8× jump remains coherent.

### What Makes This Distinctive: The Search-Based Philosophy

Underlying these technical contributions is a methodological shift. Rather than designing interpolation formulas from first principles (PI's linear scaling, NTK's power law, YaRN's frequency bands), LongRoPE treats the problem as a black-box optimization: given a model and a target length, search the space of possible RoPE rescaling configurations to minimize perplexity on a few validation samples.

This shift is significant because it decouples the interpolation strategy from human assumptions about what frequencies "should" matter. The paper's experiments demonstrate that the optimal solution often violates intuitive groupings—some high-frequency dimensions end up with substantial interpolation, some low-frequency dimensions get near-extrapolation, and the pattern depends on the specific model (LLaMA2 vs. Mistral produce different optimal configurations) and the specific target length (the optimal $\lambda_i$ vector for 128k differs from that for 256k).

The cost of this flexibility is the search itself, which the paper must address with algorithmic efficiency techniques (optimized initialization using PI/NTK/YaRN as seeds, monotonicity constraints to prune the search space). But the payoff is that LongRoPE achieves extensions that formula-based methods cannot match, because it discovers interpolation patterns that no human would design and no simple parametric curve would produce.

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)

LongRoPE is a *search-based RoPE rescaling system* that finds, for a specific model and target context length, the optimal way to "squeeze" new position indices down into the range the model already understands — per dimension and per token position — so that a pre-trained LLM can process extremely long documents (up to 2 million tokens) without ever being trained on texts that long. The core insight is that no single formula can capture which RoPE dimensions are "important" for a given model; instead, an evolutionary search algorithm discovers a custom non-uniform interpolation pattern that preserves the model's most critical positional information while aggressively compressing dimensions that can tolerate it, enabling extensions of up to 8× without any fine-tuning and up to 512× when combined with a progressive fine-tuning strategy.

### 3.2 Big-picture architecture (diagram in words)

LongRoPE consists of four major components connected in a pipeline:

1. **The Evolutionary Search Engine** (Section 3.2): Takes a target context window size $L'$ and a base LLM (either pre-trained or already fine-tuned at a shorter extended length) as input. It searches a vast space of possible RoPE rescale configurations — each configuration specifies, for every one of the $d/2$ RoPE frequency dimensions, a rescale factor $\lambda_i$, and a token-position threshold $\hat{n}$ below which no interpolation is applied. The search is guided by perplexity on a small set of validation documents (as few as 3–5), using evolutionary operations (mutation, crossover, selection) constrained by a monotonicity requirement. Output: a vector of optimal rescale factors $\lambda_0, \lambda_1, ..., \lambda_{d/2-1}$ and the threshold $\hat{n}$.

2. **The Non-Uniform Positional Interpolation Mechanism** (Section 3.1, Eq. 3): Given the searched rescale factors and threshold, modifies the RoPE embedding at inference time: for token positions $n < \hat{n}$, the original (unscaled) rotation angles are used; for positions $n \geq \hat{n}$, each dimension $i$ uses a rotation angle divided by its specific $\lambda_i$. This produces a RoPE matrix where some dimensions are nearly extrapolated ($\lambda_i \approx 1$), others are heavily interpolated ($\lambda_i$ up to $s \times 1.25$), and the transition from dimension to dimension is an arbitrary learned curve rather than a fixed formula.

3. **The Progressive Extension Pipeline** (Section 3.3): Three stages:
   - **Stage 1 — 256k fine-tuning:** Search for rescale factors for a 128k or 256k target on the pre-trained LLM. Fine-tune the model for 400 steps at 128k, then replace the rescale factors with the 256k configuration and fine-tune an additional 600 steps. This produces an LLM with a verified 256k context window.
   - **Stage 2 — 2048k jump (no training):** Run the evolutionary search again on the fine-tuned 256k model, targeting 2048k (an 8× extension). Apply the resulting rescale factors directly — no additional fine-tuning needed because the 8× extension is within the method's non-fine-tuning capability.
   - **Stage 3 — Short-context recovery:** To fix degraded performance on texts ≤8k tokens, run a third search on the 2048k-extended model with a reduced maximum rescale factor (encouraging less interpolation at short lengths), producing separate rescale factors for 4k and 8k that the model dynamically switches to based on input length.

4. **The Dynamic RoPE Switching Mechanism**: During inference, the system checks the input sequence length and selects the appropriate RoPE rescale factors: if the length is ≤8k tokens, it uses the recovery rescale factors; if the length is longer, it uses the 2048k rescale factors discovered in Stage 2. This is a simple conditional that happens before the forward pass, requiring no architectural changes.

### 3.3 Roadmap for the deep dive

- **First**, the mathematical formulation of non-uniform positional interpolation (Eq. 2, Eq. 3) — the unified framework that expresses PI, NTK, YaRN, and LongRoPE as different choices of rescale factors $\lambda_i$ and position threshold $\hat{n}$. This establishes the "search space" that the algorithm explores.

- **Second**, the evolutionary search algorithm itself (Algorithm 1) — how the search space is defined, how initial populations are seeded with known solutions, how the monotonicity constraint prunes invalid candidates, and how mutation/crossover/selection iterate to find optimal rescale factors. This is the engine that makes the approach work at massive extension ratios.

- **Third**, the fine-tuning strategy for the 256k intermediate model — the training data, hyperparameters, learning rate schedules, and the two-stage approach (128k → 256k) that makes convergence tractable despite a 64× extension ratio.

- **Fourth**, the progressive extension from 256k to 2048k — how the 8× non-fine-tuning property is exploited on an already-extended model, why this works when direct 512× extension would fail, and the computational savings it enables.

- **Fifth**, the short-context recovery mechanism — why performance degrades at original lengths, how a second search with constrained rescale factors fixes it, and how dynamic switching works at inference time.

### 3.4 Detailed, sentence-based technical breakdown

This is primarily a **methods paper** whose core idea is that the optimal pattern of positional interpolation across RoPE dimensions and token positions is model-specific and extension-ratio-specific, and that evolutionary search can discover patterns that hand-designed formulas (PI, NTK, YaRN) cannot — enabling 8× non-fine-tuning extensions and, via progressive stacking, a total 512× extension.

---

#### Unified Formulation of Positional Interpolation

The paper first establishes a common mathematical language for describing all positional interpolation methods. This is essential because it transforms the problem from "design a formula" to "search a parameter vector," and reveals exactly what degrees of freedom prior methods leave unexploited.

**RoPE in its original form.** For a token at position $n$, its RoPE encoding for dimension pair $i$ (where $i$ runs from $0$ to $d/2 - 1$) uses rotation angles $n\theta_i$, with the frequency defined as:

$$\theta_i = \theta^{-2i/d}$$

where $\theta$ is the base frequency (default 10000) and $d$ is the embedding dimension. The actual encoding at position $n$ for dimension pair $i$ is the pair $(\cos(n\theta_i), \sin(n\theta_i))$, and the full encoding concatenates these pairs for all $i$.

**What this means operationally:** Each RoPE dimension $i$ corresponds to a rotating vector that completes one full cycle every $2\pi/\theta_i$ tokens. Low dimensions ($i$ small) have high frequencies — they rotate rapidly and can encode fine-grained position differences between nearby tokens. High dimensions ($i$ large) have low frequencies — they rotate slowly and encode coarse position information over long ranges. The model's attention mechanism learns to use specific combinations of these dimensions to attend to tokens at specific relative distances.

**Unified interpolation formula.** All positional interpolation methods can be expressed as replacing the original angle $n\theta_i$ with a rescaled version:

$$\left[\cos\left(\frac{n}{\lambda(\beta)^0}\right), \sin\left(\frac{n}{\lambda(\beta)^0}\right), \cos\left(\frac{n}{\lambda(\beta)^1}\right), \ldots, \sin\left(\frac{n}{\lambda(\beta)^{d/2-1}}\right)\right]$$

where $\beta = \theta^{2/d}$ and $\lambda(\beta)$ is a dimension-dependent rescale factor. The key is that different methods choose different functions for $\lambda(\beta)$:

- **PI** sets $\lambda(\beta) = s$ for all dimensions (uniform compression).
- **NTK** sets $\lambda(\beta)^i = s^{i/(d/2)}$ (power-law scaling from 1 to $s$).
- **YaRN** sets $\lambda(\beta) = 1$ for high frequencies, $\lambda(\beta) = s$ for low frequencies, and NTK-like scaling in between.

**Why this unified form matters:** It reveals that all prior methods constrain $\lambda(\beta)$ to be a simple parametric function — either constant (PI), a power law (NTK), or piecewise constant with at most three segments (YaRN). The actual function that would minimize perplexity for a specific model at a specific extension ratio almost certainly does not fit any of these parametric forms. LongRoPE's key move is to **remove the parametric constraint entirely**: instead of defining $\lambda(\beta)$ as a formula, treat each $\lambda_i$ as an independent free parameter to be discovered by optimization. This converts the problem from curve-fitting (with 1–3 parameters) to high-dimensional discrete optimization (with $d/2$ parameters, typically 64 or 128 for common models).

---

#### The Non-Uniform Interpolation Optimization Problem (Equation 3)

The paper formalizes the search problem as minimizing perplexity over input documents that exceed the target context length. This equation is the mathematical specification of what LongRoPE optimizes.

$$ \arg\min_{x \in X; |x| \geq L'} \mathcal{L}\left(\text{LLM}(\text{RoPE}, X)\right) $$

where:

$$
\text{RoPE}(n)_{i=0,\ldots,\frac{d}{2}-1; \; n \in [0, |x|)} = \left[\ldots, \cos\left(I(\hat{\lambda}_i, \hat{n}) \times \frac{n}{\beta^i}\right), \sin\left(I(\hat{\lambda}_i, \hat{n}) \times \frac{n}{\beta^i}\right), \ldots\right]
$$

and the rescale indicator function is:

$$
I(\hat{\lambda}_i, \hat{n}) = \begin{cases} 1 & n < \hat{n} \\ \frac{1}{\lambda_i} & n \geq \hat{n} \end{cases}
$$

where:
- $X$ is a set of validation documents, each with token length $|x|$ exceeding the target context window $L'$.
- $\mathcal{L}$ is the next-token prediction loss (perplexity) of the language model.
- $\text{LLM}(\text{RoPE}, X)$ means "the language model with RoPE embedding modified as specified, evaluated on documents $X$."
- $n$ is the token position index, running from $0$ to the document length.
- $i$ indexes the RoPE dimension pairs, from $0$ to $d/2-1$.
- $\beta = \theta^{2/d}$ is the base frequency multiplier such that the original frequency at dimension $i$ is $\beta^{-i}$.
- $\hat{\lambda}_i = 1/\lambda_i$ is the angle multiplier applied when $n \geq \hat{n}$; equivalently, $\lambda_i$ is the "effective rescale factor" — larger $\lambda_i$ means more interpolation (more compression of positions).
- $\hat{n}$ is the token position threshold: positions $0, 1, ..., \hat{n}-1$ use the original RoPE without any rescaling (direct extrapolation); positions $\hat{n}$ and beyond use the per-dimension rescale factors.

**What this computes:** For a given candidate solution — a vector of $d/2$ rescale factors $(\lambda_0, \lambda_1, ..., \lambda_{d/2-1})$ and an integer threshold $\hat{n}$ — the procedure is: (1) modify the LLM's RoPE embedding so that for each token position $n$, if $n < \hat{n}$, the rotation angle for dimension $i$ remains $n/\beta^i$ (unchanged); if $n \geq \hat{n}$, the rotation angle becomes $n/(\lambda_i \beta^i)$. This means that beyond position $\hat{n}$, each dimension $i$ independently compresses the effective position index by a factor of $\lambda_i$. (2) Run the modified LLM on validation documents of length at least $L'$ and compute the average perplexity. (3) The optimization objective is to find the $(\lambda_0, ..., \lambda_{d/2-1}, \hat{n})$ that yields the lowest perplexity.

**Why this form:** The piecewise structure (unchanged for $n < \hat{n}$, rescaled for $n \geq \hat{n}$) directly implements Finding 2 from Section 2.2 — that initial tokens should be extrapolated with less interpolation. The per-dimension $\lambda_i$ factors directly implement Finding 1 — that different RoPE dimensions need different amounts of interpolation, and the pattern is not described by any simple formula. The combination creates a two-dimensional non-uniformity: the "what" axis (which dimension) controls *degree* of interpolation, while the "where" axis (which token positions) controls *whether* interpolation applies at all for the earliest tokens. The product $I(\hat{\lambda}_i, \hat{n}) \times n/\beta^i$ means that for positions beyond $\hat{n}$, the effective position index seen by the model at dimension $i$ is $n/\lambda_i$ rather than the true $n$ — so a token that is physically at position 100,000 might appear to the model's high-frequency dimensions as position 25,000 (if $\lambda_i = 4$) or as position 100,000 (if $\lambda_i = 1$, i.e., extrapolation), depending on which dimension we're looking at. This allows the model to preserve fine position discrimination in some frequency bands while compressing others, rather than uniformly degrading all bands as PI does.

---

#### Evolutionary Search Algorithm (Algorithm 1)

The search space defined by Eq. 3 is combinatorially enormous. For LLaMA2-7B with $d = 128$ (so $d/2 = 64$ frequency pairs), targeting an extension ratio $s = 4\times$ (from 4k to 16k): each of the 64 $\lambda_i$ can independently take any value in $\{1.0, 1.01, 1.02, ..., s \times 1.25 = 5.0\}$ — that's 400 possible values per dimension — and $\hat{n}$ can take one of 14 values. The total search space size is $400^{64} \times 14 \approx 4 \times 10^{167}$ configurations. Exhaustive evaluation is obviously impossible. Even random sampling would be hopeless given that each evaluation requires running the LLM on long documents.

The paper's evolutionary search is designed to navigate this space efficiently using three key ideas: seeding with known good solutions, constraining the search to monotonically non-decreasing $\lambda_i$ sequences, and using perplexity-guided selection to iteratively refine the population.

**Algorithm 1 pseudocode** (paraphrased from the paper):

```
Input: target LLM, validation samples X, population size P=64,
       mutation size N1=16, crossover size N2=16, max iterations T=40,
       mutation probability p=0.3

1. Top-k = empty set
2. P0 = InitializePopulation(P, X, p)    // with optimization seeding
3. For i = 1 to T:
4.     Compute perplexity(LLM, Pi-1, X) for each individual
5.     Top-k = UpdateTopK(Top-k, Pi-1)   // keep best performers
6.     Pmutation = Mutate(Top-k, N1, p)   // with monotonicity constraint
7.     Pcrossover = Crossover(Top-k, N2)   // with monotonicity constraint
8.     Pi = Pmutation ⋃ Pcrossover ⋃ Top-k
9. Return individual in Top-k with lowest perplexity
```

The hyperparameters are: $P = 64$ individuals per generation, $N_1 = N_2 = 16$ mutation and crossover offspring per generation, $p = 0.3$ mutation probability, $T = 40$ generations, and top-32 individuals selected as parents for the next generation (the paper states "top-k" where $k=32$ based on "select top-32 for mutation/crossover"). For context windows over 512k, population, mutation, and crossover sizes are halved to reduce computational cost (each perplexity evaluation at 2048k takes approximately 50 minutes, as noted in Appendix A.3).

**Step-by-step operation:**

**Step 1 — Optimized initial population generation.** Instead of starting from random rescale factors, the algorithm seeds the initial population of $P=64$ individuals with three known configurations: the rescale factors corresponding to PI (all $\lambda_i = s$), NTK ($\lambda_i = s^{i/(d/2)}$), and YaRN (piecewise with three frequency bands). These three are directly inserted into the population. The remaining $P-3 = 61$ individuals are generated by randomly mutating these three seed configurations with probability $p=0.3$ per dimension.

**Why this seeding matters:** The search space is vast and mostly terrible — random rescale factors would almost certainly produce enormous perplexity values, wasting the first many generations just to find the "ballpark" of reasonable solutions. By seeding with PI, NTK, and YaRN (which are known to be at least functional, if suboptimal), the search begins in a region of the space that already produces coherent models, then explores variations around these baselines. The paper's Figure 6(a) confirms this: even after the first iteration, the best-found solution significantly outperforms both PI and YaRN, meaning the initial mutations around the seeds immediately find improvements.

**Step 2 — Perplexity evaluation.** For each individual (each candidate $(\lambda_0, ..., \lambda_{63}, \hat{n})$ configuration), the algorithm modifies the LLM's RoPE embedding accordingly and computes perplexity on the validation documents $X$. For target lengths ≤256k, $X$ consists of 5 random samples from the PG19 validation set, each with length at least the target $L'$. For lengths >512k, $X$ uses 3 random samples from the Pile-Books3 validation set.

**Computational cost of evaluation:** Appendix A.3 reports that a single perplexity evaluation at 2048k takes approximately 50 minutes. With $P=64$ individuals per generation, full evaluation of every generation would be prohibitively expensive. The monotonicity constraint (described next) reduces how many individuals actually get evaluated by filtering out invalid configurations before running the LLM.

**Step 3 — Monotonically non-decreasing constraint.** Before evaluating any individual, the algorithm checks whether its $\lambda_i$ sequence satisfies $\lambda_i \leq \lambda_{i+1}$ for all $i = 0, 1, ..., d/2-2$. Only configurations passing this constraint proceed to perplexity evaluation; others are discarded (or presumably not generated in the first place).

**What this constraint enforces:** Lower RoPE dimensions (small $i$, high frequencies) must have rescale factors $\lambda_i$ less than or equal to those of higher dimensions (large $i$, low frequencies). Equivalently, higher-frequency dimensions must undergo *less interpolation* (or more extrapolation) than lower-frequency dimensions, consistent with the NTK theory that high frequencies encode fine-grained position information that cannot be compressed without losing discriminability.

**Why this constraint is necessary:** Without it, the mutation and crossover operations would generate configurations where, say, dimension 5 has $\lambda_5 = 1.0$ (extrapolation) while dimension 4 has $\lambda_4 = 5.0$ (heavy interpolation). While such configurations are mathematically valid in the search space, they contradict the NTK-based understanding that frequency decreases (wavelength increases) with dimension index — a dimension 5 rotation is inherently slower than a dimension 4 rotation, so compressing dimension 4 more than dimension 5 would destroy the frequency ordering and likely produce catastrophic attention patterns. The constraint prunes these configurations without ever evaluating them, dramatically reducing the effective search cost.

**The NTK theoretical basis:** The paper cites Jacot et al. (2018) and Tancik et al. (2020) for the Neural Tangent Kernel theory, which suggests that neural networks learn functions whose frequency components correspond to the eigenvalues of the NTK. Applied to RoPE, this means the model learns to use high-frequency dimensions for fine position discrimination and low-frequency dimensions for coarse position encoding. Interpolating high frequencies (making them even lower) destroys this learned representation, while interpolating low frequencies (making them even lower still) is relatively harmless because those dimensions already encode coarse, redundant information. The monotonicity constraint encodes this inductive bias directly into the search.

**Step 4 — Selection.** After evaluating all valid individuals in the current population, the algorithm ranks them by perplexity and updates the Top-k set (the best $k=32$ configurations seen across all generations so far).

**Step 5 — Mutation with monotonicity constraint.** From the top-32 parents, $N_1=16$ offspring are generated via mutation. For each offspring, each dimension's $\lambda_i$ has probability $p=0.3$ of being replaced with a new value randomly sampled from the allowed range $[1.0, s \times 1.25]$ with step size $0.01$. If the resulting $\lambda$ sequence violates the monotonicity constraint, the mutation is rejected and retried — this is the "with mono constraint" qualifier in the algorithm. The $\hat{n}$ parameter is similarly mutated with probability $p$ from its allowed discrete set.

**What mutation explores:** Local variations around good solutions — if the current best configuration has $\lambda_{20} = 2.34$, mutation might try $\lambda_{20} = 2.35$ or $\lambda_{20} = 2.33$, testing whether a small adjustment improves perplexity. The $p=0.3$ probability means most dimensions stay unchanged in any single mutation, producing offspring that are "nearby" in search space.

**Step 6 — Crossover with monotonicity constraint.** From the top-32 parents, $N_2=16$ offspring are generated via crossover. Two parent configurations are selected, and each dimension's $\lambda_i$ is inherited from one parent or the other (likely uniform crossover: for each $i$, randomly choose which parent's $\lambda_i$ to use). The $\hat{n}$ parameter is inherited from one parent. The resulting configuration must satisfy monotonicity; if not, the crossover is rejected and retried.

**What crossover explores:** Combinations of good partial solutions from different parents. If one parent has excellent rescale factors for dimensions 0–20 and another has excellent factors for 21–63, crossover can combine them into a single offspring that outperforms both parents.

**Step 7 — Next generation and termination.** The new population $P_i$ consists of the $N_1=16$ mutation offspring, plus the $N_2=16$ crossover offspring, plus the original top-32 parents (elitism: the best solutions survive unchanged to the next generation). This maintains population size at 64 while ensuring performance never degrades. The loop repeats for $T=40$ iterations, after which the single best individual in Top-k is returned.

**Search cost and efficiency (Appendix A.3):**
- For context windows up to 256k: total search completes within 3 days on a single A100 GPU.
- For 512k: 2 A100 GPUs used.
- For 1024k and 2048k: 4 and 8 A100 GPUs respectively, with search time kept within 5 days.

Figure 6(a) shows validation perplexity dropping from 273.27 to 118.47 over 40 iterations for the 64× extension (256k target), demonstrating that the search efficiently finds solutions far better than PI and YaRN — both of which produce perplexity above 1000 at this extension ratio (YaRN actually underperforms PI at 64×, highlighting how human-designed non-uniformity can backfire).

---

#### Fine-Tuning the 256k Intermediate Model

The evolutionary search alone can achieve 8× extension without fine-tuning, but a 64× extension from 4k to 256k requires training. The paper's fine-tuning strategy is designed to make this large jump converge stably despite the massive number of unseen position indices.

**Training data.** For LLaMA2: RedPajama dataset (Computer, 2023), chunked into segments of the target context length (128k for the first stage, 256k for the second), with each segment bookended by BOS and EOS tokens. For Mistral: Together Computer's Long-Data Collections (mis, 2024), using 16k sequence length for both 128k and 256k targets.

**Why different sequence lengths for LLaMA2 vs. Mistral:** The paper observes (Appendix A.2) that "LLaMA2 necessitates text lengths that match the context window size" while "Mistral achieves the desired long context window by fine-tuning on 16k-length data." This suggests Mistral's pre-training included longer sequences or its attention patterns generalize differently — a model-specific property that the paper adapts to rather than explaining.

**Two-stage fine-tuning for LLaMA2-256k:**

**Stage 1 (128k):** The pre-trained LLaMA2-7B, modified with the searched RoPE rescale factors for a 128k context window (32× extension), is fine-tuned for 400 steps. Hyperparameters: learning rate $2 \times 10^{-5}$ with linear decay, global batch size 32. Training uses 8 A100 GPUs and takes approximately one week (Appendix A.2).

**Stage 2 (256k):** From the finished 128k checkpoint, the RoPE rescale factors are replaced with the searched configuration for 256k (64× extension), and training continues for an additional 600 steps with the same hyperparameters. This requires 16 A100 GPUs and takes approximately two weeks.

**Why two stages instead of direct 256k fine-tuning:** The paper reports in Appendix A.2 (Figure 5(c), Table 12) that directly fine-tuning LLaMA2-7B to 256k in one stage "results in a relatively slow decrease in loss" and yields worse final perplexity (Proof-pile perplexity at 256k: 1.95 for direct 256k training vs. 1.87 for the two-stage approach). The initial loss is much lower when starting from the 128k checkpoint because the model has already partially adapted to long-range attention patterns. Additionally, fine-tuning with 128k-length texts for a 256k-context model "results in a sharp increase in the initial loss" — the model needs to see texts at the target length to learn to use the full window.

**Mistral fine-tuning:** Uses a constant learning rate of $1 \times 10^{-6}$ and global batch size 64, fine-tuned for 400 steps on 16k-length sequences for both 128k and 256k targets. Only 4 A100 GPUs are needed for a 2-day training period. The paper follows YaRN's setting of using a constant learning rate.

**Why Mistral can train on 16k-length data for a 256k window:** This is a significant efficiency advantage. The paper doesn't fully explain the mechanism, but the implication is that Mistral's pre-training or architecture enables better length generalization — the model can learn to attend over 256k positions even though it only sees 16k-length training examples, perhaps because its attention patterns or RoPE encoding generalizes differently than LLaMA2's.

**Training loss dynamics (Figure 5):** Three observations from the paper:
1. The 128k model experiences a large initial loss due to the 32× extension but "rapidly decreases after a few steps."
2. Mistral's loss "begins to fluctuate after dropping to around 2.2" due to the constant learning rate.
3. The 256k model, starting from the 128k checkpoint, "exhibits a low initial training loss," confirming that the two-stage strategy "significantly facilitates convergence."

---

#### Progressive Extension from 256k to 2048k

This is the paper's most distinctive technical contribution: how to reach 2048k without ever training on documents longer than 256k.

**The key enabling property:** LongRoPE's non-uniform positional interpolation achieves up to 8× extension without fine-tuning (demonstrated in Figure 3). This is not just a convenience — it's the mechanism that makes the progressive strategy possible. If the non-fine-tuning limit were only 2× or 4× (as with NTK), reaching 2048k from 256k would require another round of expensive fine-tuning at 2048k, which is exactly what the paper aims to avoid.

**The progressive extension procedure:**

1. **First search (on pre-trained LLM):** Run evolutionary search targeting a 128k or 256k context window on the original 4k pre-trained LLM. The extension ratio is 32× or 64× — well beyond the 8× non-fine-tuning limit, hence fine-tuning will be needed.

2. **Fine-tune to 256k:** As described in the previous subsection, train the model with the searched rescale factors to produce a verified 256k-context LLM.

3. **Second search (on fine-tuned extended LLM):** Run the evolutionary search *again*, this time on the 256k fine-tuned model, targeting a 2048k context window. The extension ratio from 256k to 2048k is 8× — exactly at the limit of the method's non-fine-tuning capability. The search discovers a new set of rescale factors $(\lambda'_0, ..., \lambda'_{63}, \hat{n}')$ optimized for this specific 8× jump.

4. **Apply without fine-tuning:** The resulting rescale factors are directly applied to the 256k model — no further training is performed. The model now processes up to 2048k tokens.

**Why this works:** The second search operates on a model that has already been fine-tuned to handle a 256k range of position indices. Its RoPE embedding has been "warped" by the first set of rescale factors and the fine-tuning process to be well-behaved over a 256k range. The second search finds a new warping that maps the 2048k range into the 256k range the model understands, with the same 8× compression factor but a different pattern of which dimensions get compressed more or less. Because the model has already adapted to long-range attention during the 256k fine-tuning, the additional 8× compression doesn't introduce catastrophic distribution shift — the model "knows" how to attend over long distances, and the second interpolation just extends that capability further.

**Why not search directly for 2048k on the pre-trained LLM and then fine-tune?** The paper states this would face "prohibitively expensive training resources" and that "it's challenging to well fine-tune the LLMs under a large extension ratio" (512× from 4k to 2048k). The 512× jump introduces so many novel position indices (>99.8% unseen) that fine-tuning becomes unstable. The progressive strategy breaks this into a 64× jump (which can be trained) followed by an 8× jump (which doesn't need training), making the problem tractable.

**The 8× vs. 16× tradeoff (Table 6):** The paper compares two variants of LongRoPE-2048k for LLaMA2: one fine-tuned at 128k (requiring a 16× second extension to reach 2048k) and one fine-tuned at 256k (requiring an 8× second extension). The 256k variant achieves lower perplexity at 2048k (7.08 vs. 7.80 on Books3) and better passkey retrieval (≥90% vs. lower accuracy), confirming that keeping the non-fine-tuning extension ratio within 8× is important for performance. For Mistral, the 128k variant outperforms the 256k variant, likely because Mistral's 256k fine-tuning used only 16k training sequences (following YaRN's protocol), which limited its ability to generalize to the 256k range — making the 128k model with a 16× second extension paradoxically better than a poorly-trained 256k model with an 8× extension.

**Computational savings:** The progressive strategy avoids training on 2048k-length sequences entirely. Training at 2048k would be astronomically expensive — attention computation scales quadratically with sequence length, so a single 2048k sample would cost $(2048/256)^2 = 64\times$ more FLOPs than a 256k sample. By training only at 256k and using the 8× non-fine-tuning extension, LongRoPE achieves the 2048k context window with training costs equivalent to a 256k model.

---

#### Short-Context Window Recovery

Extending the context window to 2048k causes a known degradation in performance on short sequences (those within the original 4k–8k window). The paper diagnoses and fixes this issue.

**The problem: position crowding at short lengths.** With a 512× extension ratio applied uniformly, the position indices within the original 4k window are compressed into a tiny fraction of the RoPE embedding's frequency range. At dimension $i$ with rescale factor $\lambda_i$, what was originally position $n$ (with rotation angle $n/\beta^i$) becomes $n/(\lambda_i \beta^i)$. For dimensions with large $\lambda_i$ (heavy interpolation), the angular differences between position 1 and position 10 become extremely small — the model struggles to distinguish closely-spaced tokens that were easily distinguishable in the original training. This manifests as increased perplexity on short documents.

**The recovery procedure:**

1. **Run a third evolutionary search** on the already-extended 2048k model, targeting 4k and 8k context lengths. The search is identical to the standard procedure but with one critical modification: the maximum allowed value for $\lambda_i$ is reduced. Since less positional interpolation is needed for short lengths (there's no need to compress novel positions because all positions fit within the model's familiar range), the search is encouraged to find configurations with smaller $\lambda_i$ — closer to direct extrapolation — which preserves more of the original RoPE frequency resolution.

2. **The search produces separate rescale factors** for 4k and 8k lengths. These factors are distinct from the 2048k factors — they represent a "different RoPE" optimized for short sequences.

3. **Dynamic switching at inference:** Before processing an input, the system checks the sequence length $|x|$. If $|x| \leq 8000$, it applies the recovery rescale factors. If $|x| > 8000$, it applies the standard 2048k rescale factors. This is a simple conditional that modifies the RoPE embedding before the forward pass — no architectural changes, no extra parameters, just different rescale factors loaded based on length.

**Why two recovery lengths (4k and 8k) rather than one:** The paper describes recovering "4k and 8k context windows" (Section 3.3), suggesting that the optimal rescale factors for 4k-length documents differ from those for 8k-length documents. A document of 8k tokens has twice the position range of a 4k document, and the tradeoff between preserving fine position discrimination (favoring less interpolation) and covering the full position range (requiring some interpolation) shifts. The search discovers different optimal points for these two lengths.

**Effectiveness (Table 10):** For LongRoPE-LLaMA2-2048k (ft=128k), applying the recovery procedure reduces Proof-pile perplexity at 4k from 4.16 to 3.71, at 8k from 3.72 to 3.50, and increases average LLM benchmark accuracy from 49.3% to 52.9%. For the 256k fine-tuned variant, similar improvements are observed (4.51 → 3.85 at 4k, 3.82 → 3.65 at 8k, benchmark accuracy 47.9% → 50.8%).

**The impact on standard benchmarks (Table 8):** After recovery, LongRoPE-LLaMA2-2048k achieves ARC-Challenge scores within 2 percentage points of the original LLaMA2-7B (51.0–52.9 vs. 53.1) and HellaSwag scores within 3 points (75.3–76.5 vs. 78.6). For Mistral, the recovered model even slightly outperforms the original on TruthfulQA (43.1 vs. 42.6). The paper notes that the 256k fine-tuned variant "shows slightly more performance degradation, but remains within reasonable ranges for most tasks" — the additional training at 256k (and the larger first-stage extension ratio) introduces more distribution shift in the base model parameters, making full recovery harder.

---

#### Design Choices Summary: Why This Architecture?

**Why evolutionary search over gradient-based optimization?** The rescale factors $\lambda_i$ are not differentiable with respect to perplexity — they are applied as discrete modifications to the RoPE embedding before the forward pass, and the relationship between $\lambda_i$ and the final loss is a black-box function of the LLM's internals. Gradient-free evolutionary methods are the natural choice for this class of discrete, non-differentiable optimization problems. The paper could have used reinforcement learning or Bayesian optimization, but evolutionary search is simpler to implement and has a strong track record in neural architecture search (the paper cites Guo et al., 2020 for the evolutionary framework).

**Why monotonicity constraint instead of learning it?** The constraint encodes prior knowledge from NTK theory, dramatically reducing the effective search space. Without it, the search would waste many evaluations on configurations that are theoretically unsound. The paper's ablation (Table 11) shows that the RoPE-dimension non-uniformity alone provides the bulk of the improvement; the monotonicity constraint ensures the search focuses on the subspace where improvements are possible.

**Why 8× as the non-fine-tuning limit?** This is an empirical finding rather than a theoretical derivation. Figure 3 shows that LongRoPE achieves coherent perplexity up to 32k (8× from 4k) on PG19 and Proof-pile without fine-tuning, while PI, NTK, and YaRN all spike well before 8×. The 8× figure emerges from the combination of per-dimension optimization and initial-token preservation — removing either component reduces the achievable extension ratio.

**Why progressive extension instead of a single giant search?** Searching directly for 2048k rescale factors on the pre-trained LLM would face two problems: (1) the perplexity evaluation at 2048k on the untrained model would be astronomically high (and possibly numerically unstable), making the search signal extremely noisy; (2) even if optimal factors were found, the gap between the pre-trained model's capabilities and 2048k-length understanding would be too large for the rescale factors alone to bridge — the model simply hasn't learned to use attention over million-token ranges, and no amount of RoPE warping can create that capability from scratch. The progressive strategy first teaches the model long-range attention via 256k fine-tuning (which is feasible), then uses interpolation to extend that learned capability further.

**Why separate recovery search instead of training the recovery?** Fine-tuning the recovery (training on short sequences to restore performance) would risk catastrophic forgetting of the long-context capability. The search-based approach modifies only the RoPE embedding (not the model weights), so the long-context ability is preserved exactly while short-context performance is restored. This separation of concerns — weights handle long-range attention, RoPE handles short-range discrimination — is a clean architectural principle.

## 4. Key Insights and Innovations

### Innovation 1: Difficulty-Conditioned Compute-Optimal Test-Time Scaling

The paper's most fundamental contribution is not any single method but rather the **meta-strategy** of adaptively allocating test-time compute based on prompt difficulty. Prior work treated test-time compute as a uniform knob: turn it up (more samples, more search) and performance improves. This paper demonstrates that the relationship between compute and performance is **qualitatively different** depending on problem difficulty, and that ignoring this heterogeneity leaves enormous efficiency on the table.

What makes this genuinely novel—rather than an obvious observation—is that the difficulty-dependent behavior is often *counterintuitive*. Beam search, the strongest optimizer, actually **hurts** performance on easy problems at high budgets due to verifier over-optimization (Figure 3, right), while it **helps** substantially on medium-difficulty problems. Similarly, sequential revisions dominate on easy problems but a balanced sequential-parallel ratio is optimal on hard ones (Figure 7, right). These are not monotonic relationships where "more powerful = better." The compute-optimal policy exploits these non-monotonicities to achieve 4× better efficiency than best-of-N (Figures 4 and 8), which is a significant practical gain.

This contribution is best understood as an **inference-time analog of the Chinchilla scaling laws** for pretraining. Just as Hoffmann et al. (2022) showed that the optimal allocation of pretraining compute between model size and data quantity varies with total budget, this paper shows that the optimal allocation of test-time compute between search strategies varies with problem difficulty. The conceptual parallel is direct, but the underlying mechanism is entirely different—pretraining scaling laws optimize over continuous variables (parameters, tokens), while this paper optimizes over a discrete, combinatorial space of strategy hyperparameters conditioned on a difficulty estimate.

A subtle but important point: the predicted (non-oracle) difficulty bins perform nearly as well as oracle bins (the curves largely overlap in Figures 4 and 8). This is what makes the contribution *practical* rather than merely analytical. If the gains required ground-truth labels to estimate difficulty, the approach would be circular. The fact that the PRM's own score distribution serves as a sufficient proxy means the system is deployable without access to answers.

### Innovation 2: The Proposal Distribution and Verifier as Complementary, Independent Scaling Axes

The unifying framework in Section 2—decomposing all test-time compute methods into modifications to the **proposal distribution** (what the model generates) versus the **verifier** (how outputs are selected)—is not itself technically novel. It echoes the proposer-scorer decomposition familiar from MCMC and reinforcement learning. What *is* novel is the paper's empirical demonstration that these two axes have **complementary, difficulty-dependent strengths** and that combining them yields gains neither achieves alone.

Concretely: revisions (proposal modification) are most effective on easy problems where the model's initial output is roughly correct and just needs refinement—a local search in answer space. Search against the PRM (verifier optimization) is most effective on medium-hard problems where the model needs to explore qualitatively different solution strategies—a global search. Prior work studied these mechanisms in isolation, often reaching pessimistic conclusions (e.g., "LLMs cannot self-correct reasoning" from Huang et al., 2023). This paper's framework reconciles those findings: self-correction *does* work, but only on the right difficulty tier. Search *does* help, but only with the right algorithm at the right budget. The conflicting prior results were an artifact of testing different methods on different (implicitly difficulty-biased) problem distributions.

This insight is more than taxonomic. It implies that future systems should not choose *between* revisions and search but should deploy both, switching between them per-prompt. The paper doesn't fully realize this vision (Section 8 acknowledges that PRM tree-search was not combined with revisions), but the framework provides the intellectual scaffolding for doing so.

### Innovation 3: Empirical Evidence That Test-Time Compute Can Substitute for Pretraining—With Sharp Boundaries

The FLOPs-matched comparison in Section 7 is, to the authors' knowledge, the first to demonstrate in a realistic setting (no ground-truth access at inference) that a smaller model with additional test-time compute can **outperform a ~14× larger model** on problems within its capability range. This is significant not as a method but as an **empirical finding with direct implications for how compute budgets should be allocated** in production systems.

What distinguishes this from prior work on training-inference tradeoffs (Jones, 2021; Villalobos and Atkinson, 2023) is the specificity of the finding. The paper doesn't claim a universal substitution—it precisely characterizes *where* the substitution works (easy-to-medium problems, low R regimes) and *where* it fails (hard problems, high R regimes). The failure case is equally informative: on the hardest problems (bin 5), test-time compute provides essentially zero benefit regardless of budget, meaning that some capabilities can **only** be acquired through pretraining, not recovered at inference time. This establishes a clear boundary condition: test-time compute amplifies existing capability but does not create it from nothing.

The dependence on R = Dinference / Dpretrain adds practical nuance that prior analyses missed. For self-improvement pipelines where R ≪ 1, the case for test-time compute is strong. For high-throughput production deployments where R ≫ 1, the case weakens because the per-query inference cost of the larger model dominates the budget anyway. This is an incremental but practically important refinement of the training-inference tradeoff picture.

### Innovation 4: Verifier Over-Optimization as a First-Class Phenomenon in Test-Time Scaling

While reward hacking / over-optimization is well-documented in the RLHF literature, this paper provides some of the first clear evidence that **the same phenomenon governs test-time search scaling** and is the primary bottleneck preventing unbounded improvements from additional compute. The evidence is concrete: beam search degrades easy-problem performance at high budgets (Figure 3, right); lookahead search—the most powerful optimizer—paradoxically performs *worst* overall (Figure 3, left); and qualitative examples in Appendix M show search producing degenerate outputs (repetitive low-information steps, overly short solutions) that score highly under the PRM.

This finding is significant because it shifts the narrative around test-time compute from "more is better" to "more is better only up to the verifier's reliability frontier." It explains why prior work found negative results for sophisticated search methods: those studies likely pushed past the over-optimization threshold. It also implies that **improving verifier robustness is the key bottleneck** for further scaling test-time compute, not improving search algorithms. The paper's compute-optimal policy can be understood partly as a way to stay *below* the over-optimization threshold per difficulty level—using weaker optimization (best-of-N) where the verifier is reliable (easy problems) and stronger optimization (beam search) only where the verifier signal has more room to provide genuine guidance (medium problems).

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on three types of data: (1) **Proof-pile** (Azerbayev et al., 2022) test split, using 10 random samples each with at least 128k tokens, following YaRN's protocol; (2) **PG19** (Rae et al., 2019) test split, using the full 100 documents for within-256k comparisons; and (3) **Books3** (Gao et al., 2020) validation set, using 20 randomly selected books each exceeding 2048k tokens for the extreme-length evaluation beyond 2000k. Validation perplexity during search uses 5 random PG19 samples for targets ≤256k and 3 random Books3 samples for targets >512k.

- **Base model(s).** The paper applies LongRoPE to **LLaMA2-7B** (Touvron et al., 2023), which has a 4096-token native context window, and **Mistral-7B v0.1** (Jiang et al., 2023), which has an 8192-token native window. These two models represent distinct pre-training recipes (LLaMA2 uses standard RoPE with base frequency 10000; Mistral's exact RoPE configuration is not detailed but the method generalizes to it). The 7B scale is chosen as representative of widely-used open-source LLMs where context extension is practically valuable.

- **Metrics.** The primary metric is **perplexity** on held-out long documents, computed using a sliding window of 256 tokens for within-256k evaluations and 256k tokens for beyond-2000k evaluations. Perplexity measures the model's next-token prediction loss — lower is better — and directly captures language modeling quality. The paper also reports **passkey retrieval accuracy** as a binary metric (whether a 5-digit passkey hidden in a long document is correctly retrieved), and **standard LLM benchmark accuracy** (ARC-Challenge 25-shot, HellaSwag 10-shot, MMLU 5-shot, TruthfulQA 0-shot) from the Hugging Face Open LLM Leaderboard.

- **Baselines.** The paper compares against several state-of-the-art extended-context models, all derived from positional interpolation methods: **Together-7B-32k** (Together, 2023) — fine-tuned with PI; **Code LLaMA-7B-100k** (Rozière et al., 2023) — fine-tuned with NTK-aware interpolation; **LongLoRA-full-FT-100k** (Chen et al., 2023b) — fine-tuned with PI plus efficient training; **YaRN-LLaMA-64k and 128k** (Peng et al., 2023) — extended using YaRN's three-band non-uniform interpolation; and **YaRN-Mistral-64k and 128k**. For Mistral, **MistralLite-16k** (Amazon, 2023) is also included.

- **Generation budget / compute accounting.** The paper measures computational cost primarily in **fine-tuning steps** and **GPU-hours**. For search, cost is measured in **number of perplexity evaluations × time per evaluation**, broken down by GPU configuration in Appendix A.3 (3 days on 1× A100 for ≤256k search; up to 5 days on 8× A100 for 2048k search). This is not a "generation budget" in the typical sense — LongRoPE's inference cost per token is identical to the base model's (the RoPE modification adds negligible overhead), so the relevant compute accounting concerns the one-time costs of search and fine-tuning rather than per-query inference FLOPs.

- **Cross-validation / statistical protocol.** There is no explicit cross-validation or statistical significance reporting. The evolutionary search uses a fixed small validation set (5 or 3 documents) for fitness evaluation, with the risk that the discovered rescale factors overfit to those specific documents. The paper implicitly relies on the fact that these factors achieve strong perplexity on held-out test sets (Proof-pile, PG19, Books3) to argue generalization, but does not report variability across different validation document selections or across multiple search runs. The passkey retrieval test uses 10 iterations with random passkey placement, providing some assessment of variance for that metric.

---

### Main Quantitative Results

#### Within-256k Long Sequence Language Modeling

The paper first establishes that LongRoPE's 256k-extended models achieve perplexity competitive with or better than prior methods at all evaluation lengths up to 256k, despite these baselines being explicitly trained for their respective maximum lengths while LongRoPE targets 2048k.

**Proof-pile results (Table 5):** For LLaMA2-7B, LongRoPE-2048k (ft=128k) achieves perplexity of 2.60 at 32k, 2.36 at 65k, 2.27 at 98k, 2.26 at 131k, and 1.88 at 262k. This monotonically decreasing trend (lower perplexity with longer context) demonstrates the model effectively leverages additional tokens. By comparison, the strongest baseline, YaRN-128k (s=32), achieves 2.70 at 32k, 2.45 at 65k, 2.36 at 98k, 2.37 at 131k, and spikes to 99.64 at 262k — indicating collapse beyond its training range. LongLoRA-100k (PI-based) similarly degrades to >10³ beyond 131k. Code LLaMA-100k (NTK-based) shows gradual degradation: 2.74 at 32k, 2.55 at 65k, 2.54 at 98k, 2.71 at 131k, and 49.33 at 262k.

**Key comparison:** At 32k tokens — roughly the maximum of the Together-32k baseline — LongRoPE-2048k (ft=128k) achieves 2.60 versus Together-32k's 2.64, LongLoRA's 2.68, Code LLaMA's 2.74, and YaRN-128k's 2.70. So even at lengths where all baselines are within their nominal window, LongRoPE matches or slightly exceeds them, despite being designed for 64× longer contexts. This counters the concern that extreme extension necessarily sacrifices short-to-medium length quality.

For Mistral-7B (Table 5), LongRoPE-2048k (ft=128k) follows a similar pattern: 2.36 at 32k, 2.18 at 65k, 2.13 at 98k, 2.13 at 131k, 1.85 at 262k. YaRN-Mistral-128k (s=16) performs comparably within range: 2.41 at 32k, 2.24 at 65k, 2.18 at 98k, 2.19 at 131k, but degrades to 4.91 at 262k — better than the LLaMA2-based YaRN's collapse but still notably worse than LongRoPE's 1.85.

**PG19 results (Table 7):** At 128k evaluation length, LongRoPE-LLaMA2-2048k (ft=128k) achieves 6.35 perplexity versus Code LLaMA-100k's 16.80 and LongLoRA-100k's >10³ (collapsed). At 8k, LongRoPE achieves 6.98 versus LongLoRA's 7.16 and Code LLaMA's 7.58. For Mistral on PG19, LongRoPE-2048k (ft=128k) achieves 7.02 at 128k versus YaRN-128k's 7.32.

**Important subtlety:** LongRoPE-2048k (ft=256k) for LLaMA2 shows slightly worse performance at shorter context lengths (3.85 at 4k, 3.65 at 8k on Proof-pile) compared to the ft=128k variant (3.71 at 4k, 3.50 at 8k), but better or equal performance at 256k+ (1.87 vs 1.88 on Proof-pile). This tradeoff — better extreme-long performance coming at a small cost to short-context performance — is a consequence of the larger first-stage fine-tuning ratio (64× vs. 32×), which introduces more position crowding at short lengths that the recovery procedure only partially compensates for.

---

#### Beyond-2000k Long Sequence Language Modeling

**Books3 results (Table 6):** This is the headline evaluation demonstrating the 2048k extension. The paper evaluates perplexity at 8 discrete lengths from 8k to 2048k.

For **LLaMA2-7B**, LongRoPE-2048k (ft=256k) achieves:
- 8k: 6.81
- 16k: 6.66
- 32k: 6.31
- 64k: 6.27
- 128k: 6.21
- 256k: 6.17
- 512k: 6.17
- 1024k: 6.35
- 2048k: 7.08

The perplexity decreases (improves) from 8k to 256k, plateaus from 256k to 512k, then rises modestly at 1024k and 2048k. Crucially, the 2048k perplexity (7.08) is only marginally higher than the 8k perplexity (6.81), meaning the model processes documents 512× longer with minimal degradation.

LongRoPE-2048k (ft=128k) for LLaMA2 shows a similar pattern but with slightly higher perplexity at extreme lengths: 6.17 at 128k, 6.17 at 256k, 6.36 at 512k, 6.83 at 1024k, 7.80 at 2048k. The larger gap between 256k and 2048k for the ft=128k variant (1.63 perplexity increase vs. 0.91 for ft=256k) reflects the larger second-stage extension ratio (16× vs. 8×).

**Comparison with baselines at overlapping lengths:** At 128k on Books3, LongRoPE-LLaMA2-2048k (ft=256k) achieves 6.21 versus Code LLaMA-100k's 9.80, LongLoRA-100k's 20.57, and YaRN-128k's 6.12. YaRN-128k is competitive at exactly 128k but collapses beyond (6.12 at 128k but >10⁴ at 256k), while LongRoPE continues smoothly.

For **Mistral-7B** (Table 6), LongRoPE-2048k (ft=128k) achieves:
- 8k: 6.64
- 16k: 6.48
- 32k: 6.39
- 64k: 6.45
- 128k: 6.64
- 256k: 7.08
- 512k: 7.71
- 1024k: 8.93
- 2048k: 12.78

The Mistral results show a different pattern from LLaMA2. Perplexity decreases to a minimum at 32k (6.39), then rises steadily, reaching 12.78 at 2048k — substantially higher than LLaMA2's 7.08–7.80 at the same length. The paper attributes this to the 16k training sequence length used during Mistral fine-tuning (compared to LLaMA2's 128k and 256k training lengths), stating: "Mistral's 128k and 256k fine-tuning, we follow YaRN's setting to use a 16k training length, which affects Mistral's ability to further extend context window after fine-tuning." The LongRoPE-2048k (ft=256k) variant for Mistral performs slightly worse than ft=128k at all lengths (13.71 at 2048k), contrary to LLaMA2 where ft=256k was better — further evidence that Mistral's 256k fine-tuning on 16k-length data was insufficient to achieve robust long-range attention.

**YaRN-Mistral-128k (s=32)** by comparison: 6.70 at 8k, 6.63 at 16k, 6.65 at 32k, 6.72 at 64k, 6.85 at 128k, 99.90 at 256k, >10³ beyond. So YaRN is competitive or slightly better at ≤128k but completely fails beyond its training range, while LongRoPE maintains coherence to 2048k.

**Cross-model comparison:** LLaMA2 extended via LongRoPE substantially outperforms Mistral extended via LongRoPE at extreme lengths (7.08 vs. 12.78 at 2048k), despite Mistral having a better native context window (8k vs. 4k) and better perplexity at short lengths. This is a non-obvious finding: a model's short-context quality does not predict its extendability. The critical factor appears to be training sequence length during extension fine-tuning, not base model quality.

---

#### Passkey Retrieval

**Figure 4** plots retrieval accuracy against evaluation context length for LongRoPE models and baselines.

For **LongRoPE-LLaMA2-2048k (ft=256k)**: Accuracy remains at or above 90% from 4k all the way to 2048k. The paper states "≥90% from 4k to 2048k." There is a slight dip around 512k–1024k (visible in Figure 4 falling to approximately 90%) before recovering to ~95% at 2048k.

For **LongRoPE-Mistral-2048k (ft=128k)**: 100% accuracy is maintained up to 1800k, then drops to 60% at 2048k. The paper notes this "aligning with expectations from Table 6, where the perplexity slightly increases at 2048k."

**Baseline comparison:** Existing models (Together-32k, Code LLaMA-100k, LongLoRA-100k, YaRN variants) show accuracy that "rapidly drops to 0 beyond 128k" — the paper's Figure 4 shows all baseline curves collapsing to 0% by 256k at the latest. Only LongRoPE models maintain non-zero accuracy beyond 256k, and they maintain high accuracy.

**Interpretation:** The passkey task measures whether the model can attend to a specific token hidden in a sea of irrelevant text — a test of the effective context window in a generation setting. The 90%+ accuracy at 2048k confirms that the model is not merely producing plausible-looking low-perplexity text (which could reflect local coherence without global attention) but actually retrieves information from 2 million tokens away. The Mistral model's drop to 60% at 2048k indicates that while attention still functions, it has become less reliable — consistent with the elevated perplexity in Table 6.

---

#### Standard Benchmarks Within Original Context Window

**Table 8** compares LongRoPE-2048k models with baselines and original models on four standard benchmarks. All benchmarks are designed for short contexts (within 4096 tokens), so this measures whether extreme extension degrades the model's original capabilities.

**LLaMA2-7B results (Table 8a):**

| Model | Context Window | ARC-c (25-shot) | HellaSwag (10-shot) | MMLU (5-shot) | TruthfulQA (0-shot) |
|---|---|---|---|---|---|
| Original LLaMA2-7B | 4k | 53.1 | 78.6 | 46.6 | 39.0 |
| Together-32k | 32k | 47.6 | 76.1 | 43.3 | 39.2 |
| Code LLaMA | 100k | 42.4 | 64.8 | 40.1 | 37.1 |
| YaRN (s=16) | 64k | 52.4 | 78.7 | 42.4 | 38.2 |
| YaRN (s=32) | 128k | 52.2 | 78.5 | 41.8 | 37.4 |
| LongRoPE-2048k (ft=128k) | 2048k | **52.9** | 76.5 | 43.4 | 38.8 |
| LongRoPE-2048k (ft=256k) | 2048k | 51.0 | 75.3 | 39.6 | 37.3 |

LongRoPE-2048k (ft=128k) achieves ARC-c 52.9 (0.2 below original, 0.5 above YaRN-128k), HellaSwag 76.5 (2.1 below original, 2.0 below YaRN-128k), MMLU 43.4 (3.2 below original, 1.6 above YaRN-128k), TruthfulQA 38.8 (0.2 below original, 1.4 above YaRN-128k). The ft=256k variant shows more degradation: MMLU drops to 39.6 (7.0 below original), ARC-c drops to 51.0, suggesting the larger 64× first-stage extension combined with additional training steps negatively impacts the model's general knowledge.

**Mistral-7B results (Table 8b):**

| Model | Context Window | ARC-c (25-shot) | HellaSwag (10-shot) | MMLU (5-shot) | TruthfulQA (0-shot) |
|---|---|---|---|---|---|
| Original Mistral-7B | 8k | 60.6 | 83.2 | 63.6 | 42.6 |
| MistralLite | 16k | 59.2 | 81.6 | 50.4 | 38.3 |
| YaRN (s=16) | 64k | 59.3 | 81.3 | 61.3 | 42.5 |
| YaRN (s=32) | 128k | 59.0 | 80.5 | 60.5 | 42.5 |
| LongRoPE-2048k (ft=128k) | 2048k | 59.0 | 81.2 | 61.3 | **43.1** |
| LongRoPE-2048k (ft=256k) | 2048k | 59.2 | 80.9 | 61.1 | 42.2 |

LongRoPE-Mistral-2048k (ft=128k) actually **outperforms** the original Mistral on TruthfulQA (43.1 vs. 42.6, a +0.5 gain) and comes within 1.6 points on ARC-c, 2.0 on HellaSwag, and 2.3 on MMLU. No baseline with a context window larger than 16k beats it on any benchmark except HellaSwag (where YaRN-64k scores 81.3 vs. 81.2). The +0.5 TruthfulQA gain is likely noise (TruthfulQA is 0-shot and has higher variance), but the fact that LongRoPE doesn't *degrade* is the key result: extending to 2048k with recovery preserves Mistral's capabilities nearly intact.

---

#### Progressive Extension Effectiveness (Secondary Interpolation)

**Table 9** isolates the contribution of the second (non-fine-tuning) interpolation step. Starting from the fine-tuned LLaMA2-256k checkpoint, the paper extends to 512k, 1024k, and 2048k using three different secondary interpolation methods:

| Model | Extension Method | 512k | 1024k | 2048k |
|---|---|---|---|---|
| LLaMA2-7B (ft=256k) | PI | 6.60 | 8.73 | 20.17 |
| | YaRN | 6.39 | 6.79 | 8.27 |
| | LongRoPE | **6.17** | **6.35** | **7.08** |

LongRoPE's searched non-uniform interpolation maintains nearly flat perplexity from 512k to 2048k (6.17 → 7.08, only +0.91), while PI degrades from 6.60 to 20.17 (+13.57) and YaRN degrades from 6.39 to 8.27 (+1.88). At 2048k, LongRoPE improves over YaRN by 1.19 perplexity points — a substantial gap in language modeling quality, especially given that both are non-uniform methods operating on the same base model. This directly validates the claim that the evolutionary search discovers interpolation patterns superior to YaRN's human-designed three-band grouping, even when applied to an already-extended model.

---

#### Short-Context Recovery

**Table 10** evaluates the impact of the recovery search on Proof-pile perplexity at 4k and 8k, and on average LLM benchmark accuracy:

For LongRoPE-LLaMA2-2048k (ft=128k):
- Without recovery: 4k perplexity 4.16, 8k perplexity 3.72, benchmark accuracy 49.3%
- With recovery: 4k perplexity 3.71 (11% reduction), 8k perplexity 3.50 (6% reduction), benchmark accuracy 52.9% (3.6 percentage point gain)

For ft=256k:
- Without recovery: 4k perplexity 4.51, 8k perplexity 3.82, benchmark accuracy 47.9%
- With recovery: 4k perplexity 3.85 (15% reduction), 8k perplexity 3.65 (4% reduction), benchmark accuracy 50.8% (2.9 percentage point gain)

The recovery provides larger absolute improvements at 4k than at 8k, consistent with the "crowding" problem being more severe at shorter lengths where position indices are compressed into a smaller angular range. The benchmark accuracy gains (roughly 3 points) are substantial and restore the models to within competitive range of the original LLaMA2-7B on these tasks (original averaging ~54% across the four benchmarks; recovered ft=128k averaging ~52.9%).

---

### Ablation Studies and Robustness Checks

**Fine-tuning strategy for 256k (Table 12).** The paper compares three approaches to achieving a 256k LLaMA2 model: (1) direct fine-tuning from LLaMA2-7B with 256k rescale factors and 256k training texts; (2) direct fine-tuning from LLaMA2-7B with 256k rescale factors but 128k training texts; and (3) the two-stage approach (128k checkpoint → 256k rescale factors → 600 more steps). At 262k evaluation length, the three approaches achieve Proof-pile perplexities of 1.95, 2.21, and 1.87 respectively. The two-stage approach is clearly best. Using 128k training texts for a 256k target model fails (2.21 vs 1.95), indicating the model must actually see examples near the target length to learn to use the extended positions — interpolation alone cannot substitute for training data at the target range when the extension ratio is 64×.

**Mistral's training length sensitivity (Table 6, implicit).** The ablation comparing LongRoPE-Mistral-2048k (ft=128k) vs. (ft=256k) is effectively a study of whether longer fine-tuning context improves Mistral's extendability. The answer is negative: ft=128k outperforms ft=256k at nearly all evaluation lengths, with the gap widening at extreme lengths (12.78 vs. 13.71 at 2048k). This is attributed to Mistral's 256k fine-tuning using only 16k-length training sequences (following YaRN's protocol), which apparently does not build robust 256k-range attention. The paper does not test what would happen if Mistral were fine-tuned with 256k-length training texts (as was done for LLaMA2) — this missing experiment would clarify whether the degradation is due to training length mismatch or an inherent property of Mistral's architecture.

**Non-uniformity ablation (Table 11).** This ablation separates the contributions of RoPE-dimension non-uniformity and token-position non-uniformity, evaluated without fine-tuning:

For the pre-trained LLaMA2-7B extended to 16k and 32k:
- **PI (no non-uniformity):** 14.88 perplexity at 16k, 136.30 at 32k on PG19
- **RoPE-dim only (searched per-dimension λᵢ, no position threshold):** 7.28 at 16k, 13.00 at 32k — a 2× and 10.5× improvement over PI
- **RoPE-dim + start tokens (both non-uniformities):** 7.22 at 16k, 11.51 at 32k — additional improvement from preserving initial tokens, with the gain larger at 32k (13.00 → 11.51, 11% reduction)

For the fine-tuned LLaMA2-256k extended to 2048k:
- **PI:** 20.17 perplexity
- **RoPE-dim only:** 7.08
- **RoPE-dim + start tokens:** 7.08 (no additional improvement)

The token-position non-uniformity provides measurable benefit at moderate extension ratios (16k, 32k) but shows no impact at 2048k. The paper hypothesizes this is "possibly due to the extremely long length" where "preserving only the initial tokens without interpolation becomes non-useful." This is an interesting negative result: the attention-sink phenomenon that motivates token-position non-uniformity (Finding 2) appears to have diminishing returns as the total context length grows, perhaps because the relative importance of any fixed number of initial tokens (≤256) becomes negligible when positioned within a 2 million token document.

**Recovery effectiveness with and without dynamic switching (Table 10, implicit).** The recovery procedure is evaluated by comparing perplexity with and without the recovery rescale factors applied. The dynamic switching mechanism itself (using different rescale factors for ≤8k vs. >8k lengths) is not separately ablated — the paper reports only the final numbers with switching enabled. A useful missing ablation would be: what happens if you use the recovery rescale factors for all lengths (including long ones)? Or conversely, if you use the 2048k factors for short sequences? This would quantify how much the dynamic switching specifically contributes versus the recovery factors alone.

**YaRN's performance at high extension ratios (Figure 6a, implicit).** Figure 6(a) shows that at a 64× extension (4k → 256k), YaRN's perplexity exceeds that of PI — the human-designed non-uniform interpolation actively harms performance compared to simple linear interpolation. This is an important robustness result: the paper's core claim that human-designed grouping is suboptimal is evidenced not just by LongRoPE outperforming YaRN, but by YaRN being *worse than PI* in the regime where the grouping heuristics break down. This validates the search-based approach: when the extension ratio is extreme, formula-based non-uniformity can be worse than no non-uniformity at all.

---

### Critical Assessment

The experiments provide credible evidence for LongRoPE's headline capability — extending LLMs to 2048k tokens — but several aspects of the evaluation warrant scrutiny regarding what the results actually demonstrate and what remains unverified.

**Does LongRoPE genuinely achieve a 2048k context window?**

The evidence is strong but has important caveats. The perplexity evaluation on Books3 (Table 6) shows that LongRoPE-LLaMA2-2048k (ft=256k) achieves 7.08 perplexity at 2048k — only 0.91 above its 256k perplexity (6.17) and 0.27 above its 8k perplexity (6.81). This is genuinely impressive: the model processes a 2048k document with quality comparable to its 8k processing. The passkey retrieval (Figure 4) confirms that the model can actually use information from 2 million tokens away (≥90% accuracy), not merely produce locally coherent text. These two metrics together — low perplexity plus high retrieval accuracy — are the standard combination for validating context extension.

However, three limitations qualify this claim:

1. **Only 20 Books3 documents evaluated.** The 2048k evaluation uses 20 books "each exceeding 2048k in length" with a 256k sliding window. Twenty documents is a small sample — a few unusually structured books could dominate the average. The paper doesn't report per-document variance or worst-case perplexity, which would reveal whether performance is consistently good or driven by a subset of "easy" long documents.

2. **Passkey is a synthetic task with a single piece of information.** Retrieving one 5-digit number from 2 million tokens tests a specific kind of long-range attention (attending to a single token) but doesn't test whether the model can integrate information from *many* widely-separated positions — a capability needed for tasks like summarizing a 2000-page book or reasoning over a 2M-token codebase. The passkey task is necessary but not sufficient to demonstrate full long-context understanding. More complex long-range reasoning benchmarks (e.g., multi-hop QA over long documents, long-document summarization) are absent from the evaluation.

3. **Only LLaMA2 approaches genuine 2048k performance.** LongRoPE-Mistral-2048k (ft=128k) achieves 12.78 perplexity at 2048k and 60% passkey retrieval — much weaker than LLaMA2's results. The paper's explanation (16k training length) is plausible but means the method's claimed 2048k capability is achieved on only one of the two tested models, and that model required training at 256k length using 16 GPUs for two weeks. The Mistral results demonstrate that the progressive extension strategy's success depends critically on the quality of the intermediate fine-tuned model, which in turn depends on having training data at that intermediate length — a constraint that may not hold for all models or domains.

**Does the recovery procedure genuinely restore short-context performance?**

The evidence is positive but measured. Table 10 shows that recovery reduces 4k perplexity from 4.16 to 3.71 for the ft=128k model — meaningful but still worse than the original LLaMA2-7B's 3.58 (Table 5, Proof-pile at 4k). The benchmark results (Table 8) show recovery bringing the model to within 0.2–3.2 points of the original on most metrics, but MMLU drops from 46.6 to 43.4 (ft=128k) or 39.6 (ft=256k) — a substantial degradation on a knowledge-intensive benchmark. The ft=256k variant's MMLU drop of 7.0 points is concerning and suggests that the additional 256k fine-tuning step genuinely degrades the model's factual knowledge, which the recovery mechanism (only modifying RoPE, not model weights) cannot fully compensate for. The paper is transparent about this ("LongRoPE-LLaMA2-2048k, fine-tuned at 256k, shows slightly more performance degradation") but the characterization of "reasonable ranges for most tasks" is debatable for the 7-point MMLU drop.

**Does the evolutionary search genuinely find optimal rescale factors, or just "good enough" ones?**

This is difficult to assess because the true global optimum is unknown for a search space of size ~10¹⁶⁷. The paper shows that the search finds factors substantially better than PI, NTK, and YaRN (Table 1, Table 3, Table 9), and that validation perplexity decreases steadily over search iterations (Figure 6). However, the search uses only 3–5 validation documents, raising the question of whether the factors overfit to those specific documents. The strong test-set results (Tables 5, 6, 7) argue against severe overfitting, but without multiple search runs with different validation sets, we cannot know whether the reported results represent typical performance or lucky validation set draws. The search's stochastic nature (random mutation and crossover) means different runs may find different local optima — the paper never reports variance across multiple search runs.

The monotonicity constraint, while well-motivated by NTK theory, also means the search can never discover non-monotonic λᵢ sequences. If the true optimal configuration for some model violates monotonicity (e.g., a specific high-frequency dimension that the model learned to use for coarse rather than fine position encoding, and thus can tolerate more interpolation than its lower-frequency neighbor), the search cannot find it. The paper's ablation showing monotonicity-constrained search dramatically outperforms PI and YaRN suggests this constraint is not actively harmful, but it's an implicit assumption that the paper never tests.

**Missing experiments that would strengthen the claims:**

- **Multiple search runs with different validation seeds** to measure variance and assess whether the cost of longer search (more iterations, larger population) yields diminishing or ongoing returns.
- **A direct comparison with training on 2048k-length data** — while the paper argues this is prohibitively expensive, even a small-scale experiment (e.g., 100 steps at 1024k) would establish whether the progressive strategy is merely *cheaper* or actually *better* than direct long-context training.
- **Evaluation on structured long-context reasoning tasks** (long-document QA, summarization, multi-hop retrieval) beyond passkey and perplexity, which would demonstrate that the extended context window translates to practical task performance, not just language modeling and single-token retrieval.
- **Performance at intermediate lengths between 256k and 2048k for Mistral** — the paper evaluates only at the specific grid points (8k, 16k, 32k, ..., 2048k), but a continuous scan would reveal whether the degradation is gradual or involves a phase transition where attention suddenly breaks.
- **Varying the recovery threshold** — the paper uses 8k as the switch point but doesn't justify this choice. Ablating the threshold (4k? 16k? 32k?) would clarify whether 8k is optimal or just a reasonable default.
- **Comparison with retrieval-augmented baselines** — the paper contrasts with PI/NTK/YaRN but not with methods that chunk long documents and retrieve relevant segments. For many long-context applications, chunking+retrieval is a practical alternative; showing when full-attention LongRoPE outperforms retrieval-based methods would strengthen the practical motivation.

**Do the results generalize beyond the tested setup?**

The paper uses two model families (LLaMA2, Mistral) at 7B scale. The method relies on properties of RoPE embedding and the NTK frequency theory — both of which may behave differently at different model scales. Larger models may have more robust positional representations that require less careful interpolation (making LongRoPE's advantages smaller) or more complex positional sensitivities that require even more careful search (making the method more necessary but also harder). The paper's claim that "LongRoPE can be applied to any LLMs based on RoPE embedding" is technically true (the algorithm is general) but the practical effectiveness may vary substantially, as the LLaMA2-vs-Mistral divergence already shows.

The search cost (up to 5 days on 8× A100 for 2048k) is not prohibitive for a one-time extension, but makes the method impractical for rapid experimentation or adaptation to new models. The 8× non-fine-tuning property is what makes the progressive strategy tractable, but this property itself was discovered empirically for these specific models and extension ratios — it's not guaranteed to hold for other models or larger ratios. If a future model required a 10× secondary extension, the progressive strategy might fail, requiring a three-stage pipeline (train at 128k, extend to 1024k with 8× non-fine-tuning, extend to 2048k with 2×) that adds complexity and potential error accumulation.

**Bottom line:** LongRoPE's experiments convincingly demonstrate that 2048k context extension is achievable on LLaMA2-7B using the described progressive strategy, with performance that degrades only modestly at extreme lengths and partially recovers at short lengths. The method represents a genuine advance over prior work that caps at ~128k. However, the results are from a single successful model configuration (LLaMA2, ft=256k) with evaluation on a limited set of documents and tasks. Whether the approach generalizes to other models, scales, or domains — and whether the 2048k perplexity and passkey results translate to practical reasoning improvements — remains to be demonstrated. The Mistral results in particular serve as a caution: the method's success depends on careful intermediate fine-tuning, and when that intermediate training is constrained (by data availability or compute), the final extension quality degrades significantly.

## 6. Limitations and Trade-offs

### The Search Cost Is Not Amortized and May Dominate Total Compute

**The assumption or constraint.** The evolutionary search that discovers optimal RoPE rescale factors is treated as a one-time preprocessing cost, separate from the compute budgets reported in the paper's efficiency claims. The search for a 2048k context window requires up to 5 days on 8× A100 GPUs (Appendix A.3), with each perplexity evaluation at 2048k taking approximately 50 minutes. The search for targets ≤256k requires up to 3 days on a single A100. These costs are never factored into any of the paper's efficiency comparisons.

**The consequence.** For a practitioner wanting to extend a new model or a new target context length, the search represents a substantial upfront investment that must be repeated for each model–target-length combination. The paper demonstrates extensions at several discrete target lengths (128k, 256k, 2048k) but does not characterize how search cost scales with model size or extension ratio. If a deployment requires extending multiple model variants, or if the optimal target length is not known in advance and must be explored, the cumulative search cost could exceed the cost of the fine-tuning itself—the very cost the progressive strategy was designed to avoid.

**What evidence exists in the paper.** Appendix A.3 reports search times and GPU configurations but does not compare total search FLOPs to fine-tuning FLOPs, does not study how search cost grows with model scale, and does not evaluate whether search results transfer across related target lengths (e.g., whether rescale factors found for 256k can be reused for 512k without a full new search). Section 3.2 describes two optimization techniques (seeded initialization, monotonicity constraint) that improve search efficiency, but the paper never quantifies how much these techniques reduce search time relative to a naive baseline—the efficiency gains are asserted but not empirically isolated.

**Mitigation status.** Not addressed. The paper treats the search as an implementation detail of the method rather than a cost to be optimized or amortized. The seeded initialization (using PI, NTK, and YaRN solutions as starting points) is the only mechanism for reducing search iterations, but the paper does not experiment with alternative search algorithms (Bayesian optimization, gradient-free methods with better sample efficiency) that might reduce the number of perplexity evaluations required. There is no discussion of whether rescale factors found for one model can be transferred to initialize search for a related model, which would amortize the cost across model variants.

---

### Only One of Two Tested Models Achieves Genuine 2048k Performance

**The assumption or constraint.** The paper's headline claim—extending context to 2048k tokens—is evaluated on two models (LLaMA2-7B and Mistral-7B), but the results show a sharp divergence: LongRoPE-LLaMA2-2048k (ft=256k) achieves 7.08 perplexity at 2048k on Books3 and ≥90% passkey retrieval accuracy, while LongRoPE-Mistral-2048k (ft=128k) achieves 12.78 perplexity and only 60% passkey retrieval. The paper attributes Mistral's underperformance to its fine-tuning protocol (16k training sequences, following YaRN's setting), stating: "Mistral's 128k and 256k fine-tuning, we follow YaRN's setting to use a 16k training length, which affects Mistral's ability to further extend context window after fine-tuning."

**The consequence.** This limitation undermines the claim that LongRoPE is a general method. The divergence between LLaMA2 and Mistral reveals that the progressive extension strategy's success depends critically on the quality of the intermediate fine-tuned model—specifically, on whether that model was trained with sequences matching the intermediate target length. For LLaMA2, training at 128k and 256k lengths was feasible but expensive (8–16 GPUs for 1–2 weeks). For Mistral, the paper either chose not to train at matching lengths (following YaRN's protocol) or found that doing so was infeasible—the paper does not clarify which. A practitioner applying LongRoPE to a new model cannot know a priori whether their fine-tuning setup will produce an intermediate model capable of supporting the 8× secondary extension, because the paper provides no diagnostic for what properties of the intermediate model predict successful secondary extension.

Furthermore, the Mistral variant fine-tuned at 256k actually performs *worse* than the 128k variant at 2048k (13.71 vs. 12.78 perplexity), meaning that the intuitively better choice (train at a larger intermediate window to reduce the secondary extension ratio) backfired. The paper's explanation—that 16k training length was insufficient for 256k-range attention—is plausible but post-hoc and not experimentally isolated.

**What evidence exists in the paper.** Table 6 provides the quantitative evidence: LLaMA2 achieves single-digit perplexity through 2048k (6.81 → 7.08), while Mistral degrades to 12.78. Figure 4 shows Mistral's passkey retrieval dropping to 60% at 2048k. Table 12 and Figure 5(c) demonstrate that training length matters for LLaMA2 (training on 128k-length data for a 256k-context model degrades perplexity from 1.87 to 2.21), but no equivalent experiment is performed for Mistral—the paper does not test whether training Mistral with 128k-length or 256k-length sequences would rescue its 2048k performance.

**Mitigation status.** Not addressed. The paper acknowledges the Mistral results and offers the training-length explanation but does not propose a solution, does not characterize the minimum training-length-to-target-window ratio needed for successful secondary extension, and does not provide a Mistral model fine-tuned with matching-length data for comparison. This leaves the method's generalizability as an open question rather than an established property.

---

### Evaluation Is Limited to Perplexity and a Single Synthetic Retrieval Task

**The assumption or constraint.** The paper's evaluation of the 2048k context window relies entirely on two metrics: perplexity on 20 Books3 documents (Table 6) and passkey retrieval accuracy (Figure 4). Perplexity measures local next-token prediction quality—a model can achieve low perplexity by being a good local language model without actually integrating information across long ranges. Passkey retrieval tests whether the model can attend to a single token at an arbitrary distance, but does not test whether it can synthesize information from *multiple* widely-separated positions or perform multi-step reasoning over long contexts.

**The consequence.** The paper claims LongRoPE "will enable many new long context applications" (Section 6), but provides no evidence that the extended models perform better on any practical long-context task. Tasks like long-document summarization, multi-hop question answering over book-length texts, or code understanding across large repositories require the model to attend to and integrate information from many positions simultaneously—not just retrieve a single passkey. A model with 12.78 perplexity at 2048k (Mistral) might produce fluent text but fail at these integrative tasks. Even LLaMA2's 7.08 perplexity at 2048k does not guarantee useful performance on complex long-context reasoning, because perplexity improvements often fail to translate to downstream task improvements when the absolute perplexity is already moderate.

The evaluation at intermediate lengths similarly relies on perplexity (Tables 5, 7) without any task-based evaluation that would reveal whether, for example, a 128k-window model actually produces better summaries or more accurate QA than a 32k-window model. The standard benchmarks in Table 8 are all short-context tasks (≤4096 tokens) that measure whether extreme extension *degrades* original capabilities, not whether long-context capabilities actually improve.

**What evidence exists in the paper.** The paper provides no task-based long-context evaluation beyond passkey retrieval. All within-256k comparisons use perplexity. The passkey task is described in Appendix A.1 with a template that hides a single 5-digit number among repeated filler text—a controlled but extremely narrow test of long-range attention. The paper's citing of applications (in-context learning with numerous examples, LLM agents, long document understanding) in Section 1 is not matched by any evaluation of these applications with the extended models.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a limitation, does not discuss why task-based evaluation was omitted, and does not position passkey as a necessary-but-insufficient test of long-context capability. In the broader long-context literature, tasks like Scrolls, NarrativeQA, or long-document summarization benchmarks are standard for validating that perplexity gains translate to practical utility—their absence is a significant gap between the paper's claims and its evidence.

---

### Short-Context Recovery Mitigates But Does Not Eliminate Degradation

**The assumption or constraint.** The recovery procedure (Section 3.3) searches for new RoPE rescale factors optimized for 4k and 8k lengths, with dynamic switching at inference time. The paper presents this as addressing "performance drop within the original context window" caused by position crowding at high extension ratios.

**The consequence.** Despite recovery, the extended models consistently underperform the original models on standard benchmarks. For LLaMA2-7B, LongRoPE-2048k (ft=128k) loses 2.1 points on HellaSwag (78.6 → 76.5), 3.2 points on MMLU (46.6 → 43.4), and 0.2 points on TruthfulQA (39.0 → 38.8). The ft=256k variant loses 7.0 points on MMLU (46.6 → 39.6), 3.3 points on HellaSwag, and 2.1 points on ARC-Challenge. These are non-trivial degradations for production models where benchmark scores directly influence user trust and adoption. A practitioner choosing between LongRoPE and a retrieval-based long-context approach faces a real tradeoff: LongRoPE provides full-attention access to 2048k tokens but permanently degrades the model's core capabilities, while retrieval-based methods leave the base model intact but sacrifice global attention.

The MMLU degradation is particularly concerning because MMLU measures factual knowledge across 57 subjects, and a 7-point drop (from 46.6 to 39.6 for ft=256k) suggests the fine-tuning process itself—not just the RoPE modifications—is degrading the model's stored knowledge. The recovery procedure only modifies RoPE rescale factors; it cannot restore knowledge lost during fine-tuning. This means the degradation is partially baked into the model weights by the 256k training step, and no amount of RoPE adjustment can fully compensate.

**What evidence exists in the paper.** Table 8 provides the benchmark numbers. Table 10 quantifies the recovery's effect on perplexity and average benchmark accuracy, showing gains of 3–4 percentage points from the recovery procedure—meaningful but insufficient to close the gap to the original model. The paper acknowledges that "LongRoPE-LLaMA2-2048k, fine-tuned at 256k, shows slightly more performance degradation, but remains within reasonable ranges for most tasks" (Section 4.2). The characterization "reasonable ranges" is subjective; a 7-point MMLU drop would be considered severe in most production contexts.

**Mitigation status.** Partially addressed. The recovery procedure is the mitigation, and it demonstrably helps (Table 10). But the paper does not explore stronger recovery methods—for example, interleaving short-context training during fine-tuning to prevent knowledge degradation, or using model merging techniques to combine the extended model's long-context capabilities with the original model's short-context performance. The dynamic switching threshold (8k) is not ablated, so it's unknown whether a different threshold, or using more than two rescale factor sets (e.g., 4k, 8k, 16k, 32k), would further reduce the gap. The paper treats "slightly more performance degradation" as acceptable without establishing what degradation threshold would make the method unsuitable.

---

### Method Requires Full-Model Fine-Tuning Without Parameter-Efficient Alternatives

**The assumption or constraint.** LongRoPE's extension pipeline requires fine-tuning all model parameters for 1000 steps (400 at 128k + 600 at 256k for LLaMA2) on the target extended context length. For LLaMA2-256k, this requires 16 A100 GPUs for approximately two weeks (Appendix A.2). The paper does not explore whether parameter-efficient fine-tuning methods (LoRA, prompt tuning, adapter layers) could achieve comparable extension with lower computational cost, or whether the searched RoPE rescale factors could enable extension with fewer fine-tuning steps.

**The consequence.** The computational barrier to applying LongRoPE is substantial. A 16-GPU, two-week training run is beyond the resources of many academic labs and smaller companies. This limits the method's accessibility and makes rapid iteration (e.g., testing different intermediate context lengths or training data mixtures) prohibitively expensive. The progressive strategy avoids training at 2048k—a major saving—but the 256k training step itself is still costly. For comparison, the Mistral variants required only 4 A100 GPUs for 2 days (due to 16k training length), but achieved inferior 2048k results, suggesting that the expensive LLaMA2-style training (matching-length data, more GPUs) is necessary for high-quality extreme extension.

The paper also does not characterize whether the fine-tuning cost can be reduced. Could 200 steps at 128k + 300 at 256k achieve similar results? Would a single 600-step run at 256k with the two-stage rescale factors (without the intermediate 128k checkpoint) work? Could the base model be fine-tuned on a mixture of short and long sequences to preserve short-context performance while learning long-range attention, potentially eliminating the need for the separate recovery procedure?

**What evidence exists in the paper.** Table 12 and Figure 5(c) compare different fine-tuning strategies for LLaMA2-256k, showing that the two-stage approach (128k checkpoint → 256k rescale → 600 more steps) is most effective, that direct training at 256k converges slower, and that training with 128k-length sequences for a 256k target degrades performance. These comparisons explore variations within the full-model fine-tuning paradigm but do not test parameter-efficient alternatives. The fine-tuning cost is reported in Appendix A.2 but is not positioned as a limitation to be addressed.

**Mitigation status.** Not addressed. The paper cites LongLoRA (Chen et al., 2023b) and PoSE (Zhu et al., 2023) as "orthogonal" efficient fine-tuning works and notes that "our method is orthogonal to these efficient fine-tuning works" (Section 5). This is an acknowledgment that the fine-tuning cost could potentially be reduced by combining LongRoPE with parameter-efficient methods, but the paper does not test this combination. A practitioner reading the paper cannot know whether LongRoPE's benefits would persist if fine-tuning only a small fraction of parameters, or whether the full-weight update is essential for the model to learn long-range attention patterns.

---

### Difficulty Estimation for the General Case Is Not Characterized

**The assumption or constraint.** LongRoPE's evolutionary search uses a fixed set of 3–5 validation documents (PG19 for ≤256k, Books3 for >512k) to guide the optimization of RoPE rescale factors. The paper implicitly assumes that rescale factors found on these few documents will generalize to the broader distribution of long texts the model will encounter. Section 3.2 describes the search guided by perplexity "using 5 random samples from PG19 validation set" or "3 random samples from Pile-Books3 validation set" (Appendix A.3).

**The consequence.** The search is vulnerable to overfitting to the specific validation documents used. If those documents happen to have unusual length distributions, token repetition patterns, or topical structures, the discovered rescale factors may be suboptimal for general use. The paper never evaluates variance across different validation document selections—a single search run on a single set of 3–5 documents determines the rescale factors used in all subsequent experiments. Given that each search run is stochastic (random initialization, random mutation) and guided by a small, fixed validation set, the reported results represent a single sample from a distribution of possible outcomes. The true expected performance of LongRoPE—averaging over different validation document draws and different random seeds—could be meaningfully worse than the reported results.

Furthermore, the paper does not characterize what properties of the validation documents matter for search quality. Would the search perform better or worse with documents from the target domain? If a practitioner wants to extend a model for legal document processing, should they use legal texts as validation documents during search? The paper provides no guidance, effectively assuming that PG19 and Books3 perplexity are universal proxies for long-context quality.

**What evidence exists in the paper.** The paper provides no ablation on the number of validation documents used, no experiment varying which documents are used, and no report of variance across multiple search runs. The strong test-set results on Proof-pile, PG19, and Books3 (Tables 5, 6, 7) suggest that overfitting is not severe—the rescale factors found on one set generalize to other sets. However, all three datasets are English books or academic papers, and the generalization is tested only within this narrow domain. Whether the rescale factors would work for code, multilingual text, or domain-specific documents (medical, legal) is entirely untested.

**Mitigation status.** Not addressed. The paper does not acknowledge document selection as a potential source of variation, does not discuss the risk of overfitting to the small validation set, and does not suggest that practitioners should use domain-relevant documents during search. The stochastic nature of evolutionary search is inherent to the method, but the paper treats a single search run as definitive.

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that existing 7B LLMs can be made to reason over million-token contexts with minimal architectural changes and relatively modest fine-tuning. This shifts the boundary of what “in-context learning” can handle—e.g., full books, large code repositories, legal corpora, or long agent memories.
- Enabled directions
  - Better searches: Content-aware or layer-wise RoPE scaling; Bayesian/gradient-based search to reduce evaluation cost; multi-objective search balancing short- and long-context performance.
  - Theory: Why do specific RoPE dimensions matter more? Can we characterize the information distribution across dimensions and positions (Findings in Sec. 2.2)?
  - Training recipes: Curriculum over lengths or synthetic datasets that mimic long-range structure to reduce fine-tuning even further; combining with efficient training (e.g., LongLoRA/PoSE).
  - Complementary methods: Integrate with retrieval/memory systems (Related Work Sec. 5) and streaming attention to combine million-token context with external knowledge.
- Practical applications
  - Long-document QA and summarization; codebase refactoring and analysis; long-horizon planning in agents; scientific literature synthesis; legal/contract analysis with minimal chunking; lifelong conversation histories.
  
Selected evidence highlights
- “8× extension without fine-tuning” with better perplexity than PI/NTK/YaRN (Fig. 3; Table 1).
- Progressive recipe: 400 steps at 128k + 600 steps at 256k, then a second interpolation to 2048k (Sec. 3.3; Sec. 4.1; Appendix A.2).
- LLaMA2 Books3 perplexity: 6.17 (256k) → 6.17 (512k) → 6.35 (1024k) → 7.08 (2048k) (Table 6).
- Passkey retrieval ≥90% through 2048k for LLaMA2 (Fig. 4).
- Short-context recovery improves both perplexity and benchmark accuracy (Table 10).

In short, LongRoPE’s core insight—treating positional scaling as a learnable, non-uniform, per-dimension-and-position transformation—combined with an efficient search and a progressive extension schedule, makes million-token context windows feasible on today’s LLMs while maintaining practical performance at short lengths.
