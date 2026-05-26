# EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty

**ArXiv:** [2401.15077](https://arxiv.org/abs/2401.15077)

## 🎯 Pitch

EAGLE introduces a novel speculative sampling framework for large language models that accelerates inference by predicting future internal features—rather than tokens—while addressing the inherent uncertainty of feature autoregression by conditioning on advanced token sequences. This allows EAGLE to achieve substantial speedups (up to 3.8x) across a range of LLMs and task types, all while provably preserving the original model's output distribution. By bypassing the need for cumbersome or ill-matched draft models and maintaining generation fidelity, EAGLE advances fast, reliable, and practical LLM deployment for quality-critical applications.

---

## 1. Executive Summary

This paper introduces **EAGLE** (Extrapolation Algorithm for Greater Language-model Efficiency), a speculative sampling framework that accelerates LLM inference by shifting the draft model's prediction target from tokens to second-to-top-layer features—autoregressing at the feature level rather than the token level—and resolving the inherent sampling uncertainty in feature prediction by conditioning on a token sequence advanced by one time step (providing the sampling outcome that determines which feature branch to follow). Evaluated across all Vicuna (7B–33B) and LLaMA2-Chat (7B–70B) models on MT-bench, HumanEval, GSM8K, and Alpaca, EAGLE achieves walltime speedup ratios of 2.7×–3.5× on LLaMA2-Chat 70B and double the throughput of vanilla decoding, representing 1.7×–2.1× speedup over Lookahead and 1.5×–1.6× over Medusa, while theoretically guaranteeing distribution-preserving generation for both greedy and non-greedy settings. The method's core insight—that feature-level autoregression is more tractable than token-level prediction but requires addressing the one-step-ahead sampling uncertainty—is validated by the finding that the shifted-token conditioning alone raises the speedup ratio from 1.9× to 2.8× on Vicuna 7B, establishing that precise feature extrapolation is achievable only when the draft model has access to the concrete token sampling decisions that disambiguate the next feature.

## 2. Context and Motivation

### The Core Problem: LLM Inference Is Bottlenecked by Autoregressive Decoding

Large language models generate text one token at a time through autoregressive decoding. For each token produced, the model must load its entire set of parameters from GPU memory, perform a forward pass through all transformer layers, compute the output distribution over the vocabulary, and sample the next token. This process is **memory-bound** rather than compute-bound: the GPU's computational units spend most of their time waiting for weights to be transferred from memory, leaving substantial computational capacity underutilized (Section 4.4). The fundamental consequence is that **generation latency scales linearly with the number of output tokens**, making LLM inference slow and expensive for any application requiring long responses—dialogue systems, code generation, document summarization, or reasoning tasks.

This problem is not merely an engineering inconvenience. It directly constrains the economic viability and user experience of deployed LLM systems. A chatbot that takes several seconds to produce each sentence of a response imposes unacceptable latency on interactive applications. An API serving millions of queries per day incurs costs proportional to the total tokens generated. In production environments where LLMs are queried continuously—customer support, coding assistants, educational tools—the inference cost often dominates the total operational expenditure, exceeding training costs for high-volume deployments. Any technique that reduces per-token latency without degrading output quality translates directly to cost savings and improved user experience.

The problem is also **theoretically fundamental**: the autoregressive assumption—that each token depends on all previously generated tokens—creates a sequential dependency that resists naive parallelization. You cannot generate the next token until you have generated all prior tokens, because each new token conditions on the full history. Breaking this sequential bottleneck without altering the model's learned distribution requires careful algorithmic design.

---

### Why Existing Acceleration Approaches Have Shortcomings

The research community has pursued several broad strategies for accelerating LLM inference, each with distinct limitations that set the stage for EAGLE.

#### Weight-Compression Methods (Quantization, Pruning, Distillation)

These approaches reduce the cost of each individual forward pass. **Quantization** (Hubara et al., 2018; Shen et al., 2020) stores model weights at lower numerical precision (e.g., int8 or int4 instead of FP16), reducing memory bandwidth requirements and enabling faster per-token computation. **Pruning** (Gale et al., 2019; Sanh et al., 2020) removes redundant weights or attention heads, creating sparser models that require fewer operations per forward pass. **Knowledge distillation** (Hinton et al., 2015) trains a smaller student model to replicate a larger teacher's behavior, trading model capacity for inference speed.

While effective, these methods are **orthogonal** to the autoregressive bottleneck itself. Reducing the cost per forward pass is valuable, but you still need one forward pass per output token. For a 100-token response, even a heavily quantized model performs 100 sequential forward passes. Quantization and pruning also inherently involve **lossy compression**—the model's output distribution may shift in ways that affect generation quality, especially at aggressive compression ratios. Distillation requires training an entirely new model, which can be prohibitively expensive (TinyLLaMA was trained on 3 trillion tokens, as the paper notes in Section 1) and may not perfectly preserve the teacher's behavior.

#### Parallel Decoding via Speculative Sampling

**Speculative sampling** (Leviathan et al., 2023; Chen et al., 2023a) represents a fundamentally different strategy. Rather than reducing per-token cost, it reduces the **number of forward passes** needed to generate a sequence of tokens. The idea is elegantly simple:

1. **Draft phase**: A lightweight "draft model" generates multiple candidate tokens quickly.
2. **Verification phase**: The target LLM processes all candidate tokens in a single parallel forward pass, computing their true probabilities.
3. **Acceptance/rejection**: Each candidate token is accepted with probability $\min(1, p_{\text{target}}(t_i) / p_{\text{draft}}(t_i))$. On the first rejection, subsequent candidates are discarded and a token is resampled from an adjusted distribution.

The mathematical guarantee (proven in Appendix A.1 of Leviathan et al., 2023) is that this procedure produces tokens distributed **identically** to sampling directly from the target LLM. The speedup comes from the fact that a single target-LM forward pass can verify (and often accept) multiple draft tokens, so the expensive target model runs less frequently.

The core challenge shifts to: **how do you obtain a draft model that is both fast and accurate?** The draft model must be substantially cheaper per forward pass than the target LLM (otherwise there's no net gain), while producing tokens that the target LLM accepts at a high rate (otherwise most of the draft computation is wasted on rejected tokens).

The paper identifies two critical limitations of existing speculative sampling approaches:

**Limitation 1: No suitable draft models for smaller LLMs.** Speculative sampling typically requires a smaller model from the same series as the draft model—for example, using LLaMA 7B as the draft model for LLaMA 70B. But what about the 7B model itself? There is no pre-existing smaller model to serve as its draft model. One could use a separately trained small model like TinyLLaMA (Zhang et al., 2024), but this fails for instruction-tuned models due to template incompatibilities between Chat and non-Chat versions. Moreover, training a dedicated draft model from scratch is prohibitively expensive: TinyLLaMA consumed 3,000B tokens of training data.

**Limitation 2: High draft model overhead diminishes gains.** Even when a smaller model exists (e.g., using 7B as draft for a 13B target), the draft model's own inference cost may be sufficiently large to cancel out the savings from reduced target-model invocations. The paper reports (Section 1, Figure 1 caption): "Employing a 7B model as the draft model for a 13B model results in slow speeds due to the high overhead of the 7B model, rendering it less efficient than vanilla autoregressive decoding." For the 33B and 70B models, speculative sampling achieved only 1.12× and 1.88× speedups, respectively—modest gains that don't justify the added complexity.

**Limitation 3: Distillation doesn't fix the bottleneck.** DistillSpec (Zhou et al., 2023) attempts to improve the draft model's acceptance rate through knowledge distillation, but the paper notes (Section 4.1) that "while distillation slightly improved the speedup ratio, the limited enhancement is because distillation aims to increase the draft model's acceptance rate, while the bottleneck for speculative sampling performance lies in the high overhead of the draft model." In other words, making the draft model more accurate doesn't help if the draft model is still too slow. The bottleneck is fundamentally about draft model **cost**, not just draft model **accuracy**.

---

#### Lightweight Drafting Without Separate Models: Medusa and Lookahead

Recent work has attempted to eliminate the separate draft model entirely by **generating drafts directly from the target LLM's internal representations**. This addresses the draft-model overhead problem: instead of running a smaller copy of the LLM, you add a small, fast module that reuses the target LLM's computed features.

**Medusa** (Cai et al., 2023) attaches multiple MLP-based "Medusa heads" to the target LLM's second-to-top-layer features. After the target LLM computes features for the current sequence, each Medusa head independently predicts a future token from those features—one head predicting the immediate next token $t_{j+1}$, another predicting $t_{j+2}$, and so on. This is extremely cheap: each head is just an MLP, and all heads run in parallel from the same feature vector. However, the **accuracy is low**—approximately 0.6 (Section 1). The independence assumption (predicting $t_{j+2}$ from $f_j$ without knowing what $t_{j+1}$ was) is a severe constraint, since in truth $t_{j+2}$ depends heavily on the intermediate token $t_{j+1}$.

**Lookahead** (Fu et al., 2023) uses n-gram matching and Jacobi iteration to generate drafts. It is confined to greedy decoding settings and achieves even lower draft accuracy than Medusa (Section 1). Both methods face a fundamental tension: they make predictions cheap by using minimal additional computation, but the resulting draft quality limits the potential speedup. Since each rejected draft token represents wasted verification computation, low draft accuracy caps the achievable acceleration.

**Crucially**, Medusa's non-greedy generation does not guarantee distribution preservation (the paper explicitly notes this in the Figure 2 caption), and Lookahead is restricted to greedy decoding. This means these methods sacrifice either the theoretical guarantee of unchanged output distribution or applicability to non-greedy (temperature > 0) generation—a significant limitation for creative or diverse generation tasks.

---

### The Two Key Observations That Motivate EAGLE

The paper's central insight emerges from analyzing **why** existing lightweight draft models (Medusa, Lookahead) achieve limited accuracy, and what structural properties of LLM feature spaces could be exploited to improve draft quality without increasing cost.

#### Observation 1: Feature-Level Autoregression Is Simpler Than Token-Level Prediction

In a standard LLM, the forward pass produces a feature vector $f_j$ at the second-to-top layer (just before the LM head). The LM head then maps $f_j$ to a distribution over the vocabulary: $p_{j+1} = \text{LM Head}(f_j)$. This means features $f_j$ contain all the information needed to predict the next token—they are, in a precise sense, an **intermediate representation that already encodes the model's prediction**.

The paper argues that autoregressing at the feature level—predicting $f_{j+1}$ from the sequence of previous features—is fundamentally simpler than autoregressing at the token level. Tokens are discrete symbols drawn from a vocabulary of tens of thousands; predicting which specific word comes next requires navigating a combinatorially large space. Features, by contrast, live in a continuous, high-dimensional space where the model's learned representations exhibit smooth structure. Sequences of features are "more regular" than sequences of tokens (Section 1). By predicting features first and then deriving tokens via the LM head (which is a simple linear transformation), the draft model can leverage this regularity.

The empirical evidence for this is presented in Figure 4 and Figure 8 (Section 4.3.2): a draft model that autoregressively predicts features achieves approximately 1.9× speedup, compared to approximately 1.5× for a draft model that predicts tokens directly—a substantial improvement from the same architecture with a different prediction target.

#### Observation 2: Feature-Level Autoregression Has Inherent Uncertainty From Sampling

While feature-level autoregression is more tractable, it encounters a complication that token-level prediction does not face: **sampling uncertainty**. In text generation, the target LLM samples a token $t_{j}$ from the distribution $p_j$. This sampling step introduces randomness—given the same feature $f_{j-1}$, the model might sample "am" or "always" to follow "I," leading to **different** feature vectors $f_j$ for each outcome.

The paper illustrates this clearly in Figure 3: starting from feature $f_I$ (corresponding to the token "I"), the next feature could be $f_{\text{am}}$ or $f_{\text{always}}$, depending on which token was sampled. A draft model that sees only $f_I$ cannot know which branch to predict, because the sampling outcome hasn't been determined yet. The feature sequence has an **inherent ambiguity** that does not exist when conditioning on tokens (where the discrete token "am" unambiguously determines the context).

This uncertainty is not just a theoretical concern—it directly limits draft accuracy. Medusa encounters the same issue when predicting spaced tokens from a single feature: from $f_I$, should the model predict the distribution that follows "am" or the distribution that follows "always"? The draft model cannot disambiguate the two possibilities without knowing the intermediate sampling outcome.

---

### EAGLE's Position: Resolving Feature Uncertainty With Shifted Tokens

EAGLE addresses both observations simultaneously:

- It **autoregresses at the feature level** (Observation 1), exploiting the greater regularity of feature sequences compared to token sequences.
- It **resolves sampling uncertainty** (Observation 2) by conditioning the feature prediction on a **token sequence advanced by one time step**—that is, it includes the already-sampled token that determines which branch of the feature tree to follow. In Figure 3's example, this means predicting $f_{\text{always}}$ by conditioning on both $f_I$ and $t_{\text{always}}$, and predicting $f_{\text{am}}$ by conditioning on both $f_I$ and $t_{\text{am}}$.

The key design choice is subtle but critical: EAGLE does not predict features from tokens alone (which would just be standard token-level autoregression). It uses **both** the feature sequence (for representational regularity) and the shifted token sequence (for disambiguation). The token sequence provides the discrete, error-free signal that resolves ambiguity about which sampling path was taken, while the feature sequence provides the rich continuous representation that makes autoregression tractable.

The paper demonstrates the impact of this design via ablation (Figure 8, Section 4.3.2):
- **feature-only** input: ≈1.9× speedup (uses regularity but suffers from uncertainty)
- **feature + shifted-token** input (EAGLE): ≈2.8× speedup (regularity + uncertainty resolved)
- **token-only** input: ≈1.5× speedup (standard autoregression, no feature-level regularity)

The shift from 1.9× to 2.8×—an increase of roughly 47%—comes purely from adding the shifted-token information, with no additional model complexity. This is the paper's central empirical finding: **the uncertainty introduced by sampling is the primary bottleneck in feature-level autoregression, and resolving it through shifted-token conditioning is both simple and highly effective**.

---

### How EAGLE Improves on Prior Speculative Sampling Methods

EAGLE distinguishes itself from prior approaches along several dimensions:

**Versus standard speculative sampling (Leviathan et al., 2023; Chen et al., 2023a):**
- **No separate draft model needed.** EAGLE's draft model is a single transformer decoder layer plus an FC layer (0.24B–0.99B trainable parameters, depending on the target model size), rather than a complete smaller LLM. This eliminates the problem of finding suitable draft models for smaller target LLMs (e.g., 7B models) and drastically reduces draft-model overhead.
- **Reuses target LLM components.** The embedding layer and LM head are borrowed directly from the frozen target LLM, requiring zero additional training for these components and ensuring perfect compatibility with the target model's representations.

**Versus Medusa (Cai et al., 2023):**
- **Higher draft accuracy.** EAGLE achieves approximately 0.8 acceptance rate versus Medusa's 0.6 (Section 1), because EAGLE's autoregressive feature prediction captures sequential dependencies that Medusa's independent per-token prediction misses.
- **Distribution preservation in non-greedy settings.** Medusa's non-greedy generation does not guarantee lossless distribution preservation. EAGLE maintains the theoretical guarantee from speculative sampling for both greedy and non-greedy settings, because it uses the same verification procedure (the acceptance/rejection mechanism is unchanged). This is explicitly noted in the Figure 2 caption.
- **Conditioning on intermediate sampling outcomes.** Medusa predicts $t_{j+1}, t_{j+2}, \ldots$ all from the same feature vector $f_j$, without knowing the intermediate tokens. EAGLE conditions each feature prediction on the previous token, resolving the ambiguity that limits Medusa's accuracy.

**Versus Lookahead (Fu et al., 2023):**
- **Applicable to non-greedy decoding.** Lookahead is restricted to greedy settings. EAGLE supports both greedy (temperature=0) and non-greedy (temperature=1) generation with the same theoretical guarantees.
- **Higher draft accuracy.** Lookahead's n-gram and Jacobi-based drafts achieve even lower accuracy than Medusa (Section 1), while EAGLE's feature-level autoregression achieves substantially higher acceptance rates.

**Versus all prior lightweight draft methods:**
- **Low training cost.** EAGLE trains on 68,000 dialogues from ShareGPT (approximately 2–4B tokens) in 1–2 days on 4× A100 GPUs for the 70B model, or on a single RTX 3090 node for 7B/13B models. This contrasts sharply with training a separate draft model from scratch (3,000B tokens for TinyLLaMA).
- **Fixed dataset, not target-LLM-generated data.** EAGLE uses a pre-existing dataset (ShareGPT) rather than requiring the target LLM to generate training data autoregressively. Ablation (Section 4.3.3) shows minimal sensitivity to this choice—using target-LLM-generated data improves the speedup ratio only slightly (2.78× → 2.88× on LLaMA2-Chat 7B), confirming that EAGLE does not require the expensive step of generating on-policy training data from the target model.
- **Theoretically guaranteed distribution preservation.** Unlike Medusa and Lookahead, EAGLE's verification procedure is standard speculative sampling, which provably maintains the target LLM's output distribution exactly (Leviathan et al., 2023, Appendix A.1). The paper is explicit: "EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings" (Section 1).

---

### The Practical Significance: Why 2.7×–3.5× Matters

The paper's reported speedup ratios—2.7×–3.5× on LLaMA2-Chat 70B—are significant for several practical reasons:

**Cost reduction at scale.** For production LLM systems processing millions of queries per day, a 3× speedup means either serving the same load with one-third the GPU capacity, or serving three times the traffic with the same infrastructure. Since inference dominates operational costs for high-volume deployments, this translates to substantial cost savings.

**Latency improvements for interactive applications.** A query that takes 3 seconds with vanilla decoding takes 1 second with EAGLE—crossing a threshold where interaction feels responsive rather than sluggish. This directly impacts user experience and retention.

**Compatibility with orthogonal methods.** EAGLE operates in parallel with quantization, compilation, and other acceleration techniques. Section 4.2 demonstrates combining EAGLE with gpt-fast (PyTorch's compilation + quantization framework) to achieve 160.4 tokens/s on LLaMA2-Chat 7B on a single RTX 3090 GPU—approximately 6.5× faster than vanilla HuggingFace decoding (24.5 tokens/s). This composability means EAGLE's gains multiply with improvements in lower-level inference infrastructure.

**Enabling smaller-model deployment.** Since EAGLE does not require a separate draft model, it makes speculative sampling viable for smaller LLMs like 7B models, where no suitable pre-existing draft model exists. This extends the applicability of speculative sampling to edge devices and resource-constrained environments where only a single model can be loaded.

**Amortized training cost.** EAGLE requires a single training session that is then reused for all queries. As the number of queries grows, the amortized training cost per query approaches zero. For a production system handling millions of queries, the 1–2 day training investment is negligible compared to the ongoing inference savings.

---

### The Paper's Framing: Rethinking Where Prediction Happens

The paper positions its contribution not as an incremental improvement to speculative sampling, but as a **rethinking of what the draft model should predict and what information it needs to do so**. The title itself—"Speculative Sampling Requires Rethinking Feature Uncertainty"—emphasizes that the core contribution is conceptual: identifying that feature-level autoregression is the right level of abstraction for draft models, but that it only works if you explicitly address the uncertainty created by the sampling process.

This framing explains the paper's structure: the two observations (Section 1) are presented as the primary intellectual contribution, with the EAGLE architecture (Section 3) serving as a concrete instantiation. The ablation study (Section 4.3.2, Figure 8) is the critical evidence, because it isolates the effect of each ingredient—feature-level autoregression and shifted-token conditioning—and shows that both are necessary for the full speedup. The method's other components (tree attention, combined loss function, noise augmentation) are presented as engineering refinements that improve on the core idea but are not the central insight.

This conceptual contribution distinguishes the paper from prior work that primarily focused on architectural innovation (Medusa's multiple heads, Lookahead's Jacobi iteration) without analyzing the fundamental properties of the prediction problem itself. By identifying **why** feature-level prediction underperforms (sampling uncertainty) and providing a targeted solution (shifted-token conditioning), the paper establishes a principled basis for future improvements rather than just a new set of hyperparameters to tune.

## 3. Technical Approach

### 3.1 Reader Orientation

**What the system is:** EAGLE is a plug-in acceleration module for autoregressive language models that sits alongside the frozen target LLM and generates draft token sequences for speculative sampling verification, without modifying the target model's weights or output distribution. **What problem it solves and the shape of the solution:** EAGLE addresses the draft model bottleneck in speculative sampling—existing draft models are either too slow (separate smaller LLMs) or too inaccurate (lightweight MLP heads)—by autoregressively predicting second-to-top-layer features instead of tokens, and crucially, by conditioning each feature prediction on the already-sampled token from one time step ahead, which disambiguates the feature trajectory that would otherwise be uncertain due to the randomness of token sampling.

### 3.2 Big-Picture Architecture (Diagram in Words)

EAGLE's inference pipeline has four major components:

1. **Frozen Target LLM (Vicuna, LLaMA2-Chat, or Mixtral):** Produces token probabilities and second-to-top-layer features during both the initial prompt processing and the verification phase. Its weights are never modified.

2. **Draft Model (Autoregression Head):** A lightweight trainable module—one FC layer plus one transformer decoder layer—that takes as input a sequence of features from the target LLM and a sequence of tokens advanced by one time step, and predicts the next feature vector. It reuses the target LLM's frozen Embedding layer and LM Head.

3. **Tree-Attention Drafting Mechanism:** Generates a tree-structured draft (multiple candidate tokens at each position) through multiple autoregressive forward passes of the draft model. The tree structure—wider and deeper for higher-probability branches—enables generating more candidate tokens per forward pass than a linear chain.

4. **Speculative Verification Procedure:** The target LLM processes the entire draft tree in one parallel forward pass, computing true probabilities for all nodes. Multi-round speculative sampling (Algorithm 1 in the paper) recursively accepts or rejects tokens, guaranteeing that the final output distribution matches vanilla autoregressive decoding exactly.

Information flows as follows: the user provides a prompt → the target LLM processes the prompt and produces features for each token → the draft model takes these features plus the shifted token sequence to autoregressively predict next features → the LM Head converts predicted features to token distributions → a tree of candidate tokens is built through multiple draft-model forward passes → the target LLM verifies all tree nodes in one parallel forward pass → accepted tokens and their features are appended to the context → the cycle repeats for the next drafting and verification round.

### 3.3 Roadmap for the Deep Dive

- **First**, the core drafting mechanism—how the draft model predicts features autoregressively, why features rather than tokens, and how the shifted-token input resolves sampling uncertainty. This is the central innovation and everything else builds on it.

- **Second**, the draft model architecture in detail—the Embedding layer, FC layer, decoder layer, LM Head, and how they compose to transform (feature sequence, shifted-token sequence) inputs into next-feature predictions.

- **Third**, the training procedure and loss functions. EAGLE has a combined regression + classification objective, uses noise augmentation on features, and trains on a fixed dataset rather than target-LLM-generated data. Understanding the training choices explains why EAGLE achieves high accuracy with low training cost.

- **Fourth**, the tree-structured draft generation mechanism—how tree attention enables producing many candidate tokens from few draft-model forward passes, and the specific branching structure used.

- **Fifth**, the verification phase—how multi-round speculative sampling on tree-structured drafts ensures distribution preservation, and how the recursive acceptance/rejection algorithm differs from standard chain-structured speculative sampling.

- **Sixth**, the broader inference loop that alternates between drafting and verification, reusing accepted features from the verification pass as inputs to the next drafting round.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **method paper** whose core idea is that feature-level autoregression, when combined with shifted-token conditioning to resolve sampling uncertainty, produces a draft model that is simultaneously fast (one lightweight decoder layer), accurate (approximately 0.8 acceptance rate), and theoretically sound (lossless distribution preservation via standard speculative sampling verification).

---

#### The Central Mechanism: Feature-Level Autoregression With Shifted-Token Conditioning

**What the draft model predicts.** Unlike standard speculative sampling where the draft model directly predicts tokens (discrete symbols from a vocabulary), or Medusa where independent MLP heads predict future tokens from a single feature vector, EAGLE's draft model predicts the **second-to-top-layer hidden state** of the target LLM—the feature vector that sits immediately before the LM head. For a given target LLM, let $f_i$ denote this feature vector at position $i$ in the sequence (the output of the final transformer layer before the LM head projection). The feature $f_i$ has dimension `hidden_dim` (e.g., 4096 for LLaMA 7B). The relationship between features and tokens is:

$$p_{i+1} = \text{LM Head}(f_i)$$

where $p_{i+1}$ is the probability distribution over the vocabulary for the next token at position $i+1$, and LM Head is a (typically linear) transformation that maps from `hidden_dim` to vocabulary size. The next token $t_{i+1}$ is then sampled from $p_{i+1}$.

EAGLE's draft model takes as input the feature sequence $F_{1:i}$ and the token sequence $T_{2:i+1}$ (advanced by one time step relative to the features) and predicts $\hat{f}_{i+1}$, the feature that would correspond to token $t_{i+1}$:

$$\hat{f}_{i+1} = \text{DraftModel}(F_{1:i}, T_{2:i+1})$$

where $F_{1:i} = (f_1, f_2, \ldots, f_i)$ is the feature sequence produced by the target LLM for tokens $t_1$ through $t_i$, and $T_{2:i+1} = (t_2, t_3, \ldots, t_{i+1})$ is the token sequence advanced by one position—the token at position $j+1$ in this sequence is the token that was actually sampled (or will be sampled) at position $j+1$ in the target LLM's autoregressive process.

**Why features are easier to predict than tokens.** The paper argues that feature sequences exhibit greater regularity than token sequences. Tokens are discrete symbols drawn from a vocabulary of tens of thousands (32,000 for LLaMA models). The mapping from context to next token is a classification problem over this large discrete space, where semantically similar tokens (e.g., "happy" and "joyful") receive distinct categorical identities despite their conceptual proximity. Feature vectors, by contrast, live in a continuous high-dimensional space where the target LLM's learned representations impose smooth structure—sequences of features follow trajectories through this space that reflect the underlying semantics and syntax, with nearby features corresponding to similar continuation contexts.

The draft model exploits this regularity by learning to predict the next feature $\hat{f}_{i+1}$ from the preceding feature sequence. Once $\hat{f}_{i+1}$ is predicted, the next token's distribution is recovered by applying the target LLM's frozen LM Head:

$$\hat{p}_{i+2} = \text{Softmax}(\text{LM Head}(\hat{f}_{i+1}))$$

This decomposition—predict a continuous feature vector, then map to discrete tokens via the LM Head—is what makes the prediction task more tractable than direct token prediction. The feature prediction is a regression problem in a continuous space; the token prediction is a classification problem over a discrete vocabulary. The draft model only needs to learn the feature dynamics, not the mapping from features to tokens, because the LM Head already encodes that mapping perfectly (and is reused without modification).

Figure 4 and Figure 8 in Section 4.3.2 provide the empirical validation: a draft model predicting features achieves approximately 1.9× speedup on Vicuna 7B, compared to approximately 1.5× for a structurally identical draft model that predicts tokens directly. The speedup ratio reflects both higher acceptance rates (more draft tokens accepted per verification pass) and the same draft-model cost.

**The sampling uncertainty problem.** Feature-level autoregression encounters a complication: in standard autoregressive text generation, the next token $t_{i+1}$ is **sampled** from the distribution $p_{i+1}$ (whether greedily at temperature 0 or stochastically at temperature > 0). This sampling step introduces randomness—given the same feature $f_i$, different random draws will produce different tokens, and consequently different feature vectors $f_{i+1}$.

Consider the paper's example (Figure 3): starting from the token "I" with feature $f_I$, the model's distribution over next tokens includes both "am" (probability 0.6) and "always" (probability 0.4). If the model samples "am," the subsequent feature will be $f_{\text{am}}$; if it samples "always," the subsequent feature will be $f_{\text{always}}$. A draft model that sees only $f_I$ faces an ambiguous prediction problem—it must predict the next feature without knowing which branch of the sampling tree was taken. This is not merely a noise issue but a structural ambiguity: the feature $f_{\text{am}}$ and $f_{\text{always}}$ are different vectors in the hidden space, and there is no deterministic mapping from $f_I$ alone to either one.

The paper formalizes this as **feature uncertainty**: the mapping from $f_i$ to $f_{i+1}$ is not a function but a one-to-many relation, because the intermediate sampling step determines which of several possible next features actually occurs. This uncertainty fundamentally limits the accuracy of feature-level autoregression. A draft model trained to minimize expected prediction error will learn to predict some average of the possible next features, which may not correspond to any actual feature that the target LLM produces.

**How shifted tokens resolve the uncertainty.** EAGLE's key insight is that including the **already-sampled token** $t_{i+1}$ as an input to the draft model resolves the ambiguity. The token $t_{i+1}$ encodes the outcome of the sampling step—it tells the draft model which branch of the feature tree was actually taken. Given both $f_i$ and $t_{i+1}$, the mapping to $f_{i+1}$ becomes deterministic (for a given target LLM, the feature for a specific token in a specific context is uniquely determined).

The draft model therefore receives a **token sequence advanced by one time step** relative to the feature sequence. When predicting $\hat{f}_{i+1}$, it sees:
- Features $f_1, f_2, \ldots, f_i$ (the target LLM's internal representations up to position $i$)
- Tokens $t_2, t_3, \ldots, t_{i+1}$ (the actual sampled tokens from positions 2 through $i+1$)

The token $t_{i+1}$ is the one that was sampled following feature $f_i$—the critical piece of information that tells the draft model "we went down the 'am' branch, not the 'always' branch."

During the drafting phase, the draft model accesses these shifted tokens because the drafting process generates tokens autoregressively. When the draft model produces its first prediction for position $i+1$, it has access to the ground-truth tokens $t_2, \ldots, t_i$ (which have already been generated and verified) plus the token $t_{i+1}$ that was just sampled from the previous draft-model prediction. For subsequent draft steps, the draft model uses its own previously predicted (and sampled) tokens. This means the draft model always has access to the concrete token decisions that determine which feature trajectory it should follow.

The ablation study in Figure 8 (Section 4.3.2) demonstrates the magnitude of this effect. On Vicuna 7B at temperature 0, the speedup ratios for different draft-model inputs are:
- **feature-only** (no tokens): approximately 1.9× speedup
- **feature + unshifted-token** (tokens aligned with features): approximately 2.1× speedup
- **feature + shifted-token** (EAGLE): approximately 2.8× speedup
- **token-only**: approximately 1.5× speedup

The jump from 1.9× to 2.8× when adding shifted-token information—a 47% improvement—comes from purely resolving the sampling uncertainty, with no increase in model capacity or computational cost. The feature+unshifted-token variant provides only a modest improvement over feature-only (+0.2× speedup), which the paper attributes primarily to the discrete, error-free tokens helping mitigate error accumulation in the feature autoregression rather than resolving the fundamental ambiguity. Only the shifted tokens provide the sampling outcome information needed to disambiguate the next feature.

**Why the shifted-token approach works in practice despite being autoregressive.** A potential concern is that conditioning on $t_{i+1}$ seems circular—to predict $f_{i+1}$ you need to know $t_{i+1}$, but $t_{i+1}$ is what you're ultimately trying to generate. The resolution is that during drafting, $t_{i+1}$ is not the ground-truth token from the target LLM but rather the token that the draft model itself sampled at the previous step. EAGLE's drafting process works as follows:

1. **Initial step:** The draft model has access to $F_{1:i}$ (ground-truth features from the target LLM's verification of previously accepted tokens) and $T_{2:i+1}$ (ground-truth tokens for positions 2 through $i$, plus $t_{i+1}$ which was sampled during verification). It predicts $\hat{f}_{i+1}$.
2. **Apply LM Head:** $\hat{p}_{i+2} = \text{Softmax}(\text{LM Head}(\hat{f}_{i+1}))$, and sample $\hat{t}_{i+2}$ from $\hat{p}_{i+2}$.
3. **Next draft step:** Append $\hat{f}_{i+1}$ to the feature sequence and $\hat{t}_{i+2}$ to the token sequence, forming $F_{1:i+1}$ and $T_{2:i+2}$. Predict $\hat{f}_{i+2}$.
4. **Continue autoregressively** for the desired draft length.

The shifting means that at each draft step, the draft model conditions on the token it just sampled—closing the loop and allowing the feature prediction to track the specific sampling path being generated. The first step in the chain uses an actual $(f_i, t_{i+1})$ pair from the verification phase, which provides an anchor of ground-truth conditioning before the draft model continues under its own predicted features.

---

#### Draft Model Architecture

The EAGLE draft model comprises three modules, two of which are borrowed frozen from the target LLM and one of which is the only trainable component (Section 3.1, Figure 6).

**Embedding Layer (frozen, from target LLM).** The token embedding layer from the target LLM maps each discrete token $t_j$ in the shifted token sequence to a continuous embedding vector $e_j$ of dimension `hidden_dim`. This is the same embedding table used during the target LLM's own forward passes. Reusing it ensures that the token representations seen by the draft model are exactly the representations the target LLM uses internally, eliminating any embedding mismatch that could degrade feature prediction accuracy.

**LM Head (frozen, from target LLM).** After the draft model predicts a feature vector $\hat{f}_{i+1}$, the target LLM's LM Head maps it to a logit vector over the vocabulary, and softmax produces the probability distribution $\hat{p}_{i+2}$. The draft model itself never learns to map features to tokens; it relies entirely on the pre-trained mapping that already exists in the target LLM. This has two advantages: (1) it eliminates the need to learn a vocabulary-sized classification head, drastically reducing the draft model's parameter count, and (2) it guarantees that the draft model's token distributions are computed using the same feature-to-token mapping that the target LLM uses, maximizing the alignment between draft and target probabilities (which improves acceptance rates).

**Autoregression Head (trainable).** This is the only component of EAGLE that requires training. It consists of two sub-components:

**FC Layer (dimensionality reduction):** The input to the draft model at each position is a concatenation of the feature vector $f_j$ and the token embedding $e_{j+1}$ (the embedding of the shifted token). Both have dimension `hidden_dim`, so the concatenated vector has dimension `2 × hidden_dim`. The FC layer projects this down to `hidden_dim`:

$$h_j = \text{FC}([f_j; e_{j+1}])$$

where $[f_j; e_{j+1}]$ denotes concatenation, and FC is a learned linear transformation (a weight matrix and bias). This reduction is purely for computational efficiency—passing a `2 × hidden_dim` vector into the decoder layer would increase its parameter count and computational cost. The FC layer learns to fuse the feature and token information into a single `hidden_dim` representation that the decoder layer can process with its existing architecture.

**Transformer Decoder Layer:** A single standard transformer decoder layer processes the sequence of fused representations $h_1, h_2, \ldots, h_i$. The decoder layer includes self-attention (allowing each position to attend to all previous positions in the input sequence), followed by a feed-forward network, with residual connections and layer normalization. The output at the final position $i$ is the predicted next feature:

$$\hat{f}_{i+1} = \text{DecoderLayer}(h_{1:i})[i]$$

where DecoderLayer$(h_{1:i})$ produces a sequence of outputs of the same length, and $[i]$ indexes the final output position. The decoder layer's self-attention mechanism allows the prediction of $\hat{f}_{i+1}$ to incorporate information from the entire history of features and shifted tokens, not just the immediate predecessor. This is crucial because the next feature depends on the full context—not just the most recent feature and token, but potentially earlier ones as well.

**Why a single decoder layer?** The paper's design rationale is that the draft model should be as lightweight as possible while still being accurate enough to produce high-acceptance-rate drafts. A single decoder layer with `hidden_dim` matching the target LLM's hidden dimension (e.g., 4096 for 7B models, 8192 for 70B models) provides sufficient capacity to learn the feature-level autoregressive dynamics while keeping the draft model's latency orders of magnitude lower than the target LLM's (which has 32–80 layers). The paper reports trainable parameter counts of 0.24B (7B models), 0.37B (13B models), 0.56B (33B models), and 0.99B (70B models)—all substantially smaller than even the smallest target LLM.

**The full forward pass of the draft model.** Putting it together, at inference time the draft model executes the following computation for a sequence of length `seq_len`:

1. **Token embedding lookup:** For each position $j$ from 1 to `seq_len`, retrieve $e_{j+1}$ from the frozen Embedding layer for the shifted token $t_{j+1}$. The embedding sequence has shape `(bs, seq_len, hidden_dim)`.

2. **Concatenation:** At each position, concatenate $f_j$ (from the target LLM's features, shape `(bs, seq_len, hidden_dim)`) with $e_{j+1}$ (shape `(bs, seq_len, hidden_dim)`) to form a fused input of shape `(bs, seq_len, 2 × hidden_dim)`.

3. **Dimensionality reduction:** Apply the FC layer to project each position's concatenated vector down to `(bs, seq_len, hidden_dim)`.

4. **Autoregressive processing:** Pass the projected sequence through the transformer decoder layer with causal self-attention (each position attends only to itself and previous positions, not future ones). This produces an output tensor of shape `(bs, seq_len, hidden_dim)`.

5. **Feature extraction:** Take the output at the last position (or, during tree-structured drafting, at the positions where new predictions are needed) as the predicted next feature $\hat{f}_{i+1}$ of shape `(bs, hidden_dim)`.

6. **Token distribution computation:** Apply the frozen LM Head to $\hat{f}_{i+1}$ to obtain logits, then softmax to obtain $\hat{p}_{i+2}$, the predicted probability distribution over the vocabulary.

7. **Token sampling:** Sample $\hat{t}_{i+2}$ from $\hat{p}_{i+2}$ (greedy argmax at temperature 0, or multinomial sampling at temperature > 0).

At the next draft step, $\hat{f}_{i+1}$ is appended to the feature sequence and $\hat{t}_{i+2}$ is appended to the token sequence (as the shifted token for position $i+2$), and the process repeats.

---

#### Training the Draft Model

EAGLE's Autoregression Head (FC layer + decoder layer) is the only component that undergoes training. The Embedding layer and LM Head remain frozen, and the target LLM's weights are never modified. Section 3.2 describes the training procedure in detail.

**Training data construction.** The training data consists of dialogue examples from the ShareGPT dataset—68,000 multi-turn conversation iterations (Section 4). For each training example, the target LLM is frozen and used to compute features for the entire conversation. Specifically:

1. A conversation (sequence of user and assistant turns) is fed through the target LLM.
2. For each token position $i$ in the response portion, the target LLM produces the feature $f_i$ (the second-to-top-layer hidden state) and the token $t_i$.
3. The training sample for the draft model is created as: input = feature sequence $F_{1:i}$ + shifted token sequence $T_{2:i+1}$, target = feature $f_{i+1}$. This requires only a single forward pass of the target LLM per conversation to extract all features; no autoregressive generation from the target LLM is needed for data creation.

This approach uses a **fixed dataset** (ShareGPT) rather than generating on-policy data from the target LLM autoregressively. The paper argues this is a deliberate design choice to minimize training overhead—having the target LLM generate its own training data would require running it repeatedly in autoregressive mode, which is expensive. Section 4.3.3 shows that using target-LLM-generated data improves the speedup ratio only slightly (from 2.78× to 2.88× on LLaMA2-Chat 7B), confirming that EAGLE's draft model is not highly sensitive to training data distribution.

**Combined regression and classification loss.** The draft model's primary training objective is to predict the correct next feature $f_{i+1}$—a regression task. However, the ultimate goal is not feature prediction accuracy per se, but token prediction accuracy (since the draft tokens are what get verified against the target LLM). The paper therefore introduces a combined loss that jointly optimizes for both feature-level regression and downstream token-level classification.

**Regression loss (Smooth L1):**

$$\mathcal{L}_{\text{reg}} = \text{SmoothL1}(f_{i+1}, \text{DraftModel}(T_{2:i+1}, F_{1:i}))$$

where $f_{i+1}$ is the ground-truth feature from the target LLM at position $i+1$, and $\text{DraftModel}(T_{2:i+1}, F_{1:i})$ is the draft model's predicted feature $\hat{f}_{i+1}$.

**What Smooth L1 computes:** For each element of the feature vector, Smooth L1 applies a squared loss when the absolute error is small (less than a threshold $\beta$, typically $\beta=1$) and a linear loss when the absolute error is large:

$$\text{SmoothL1}(x, y) = \begin{cases} 0.5(x - y)^2 / \beta & \text{if } |x - y| < \beta \\ |x - y| - 0.5\beta & \text{otherwise} \end{cases}$$

The result is a scalar loss value averaged over all elements of the feature vector.

**Why Smooth L1 over MSE or L1:** Standard Mean Squared Error (MSE) loss would heavily penalize large prediction errors (due to the quadratic term), which can cause training instability and make the model overly conservative—it might learn to predict average features that are always somewhat wrong rather than making confident predictions that are occasionally far off. Standard L1 loss has constant gradients regardless of error magnitude, which can lead to slow convergence for small errors. Smooth L1 combines the benefits: quadratic behavior for small errors (efficient gradient-based optimization near the optimum) and linear behavior for large errors (reduced sensitivity to outliers). For feature prediction, where the hidden states can have element values ranging over multiple orders of magnitude, robustness to outlier dimensions is important.

**Classification loss (cross-entropy on token distributions):**

Once the draft model predicts $\hat{f}_{i+1}$, the frozen LM Head produces a token distribution $\hat{p}_{i+2} = \text{Softmax}(\text{LM Head}(\hat{f}_{i+1}))$. The ground-truth distribution $p_{i+2}$ is obtained by applying the LM Head to the true feature $f_{i+1}$: $p_{i+2} = \text{Softmax}(\text{LM Head}(f_{i+1}))$. The classification loss is then:

$$\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(p_{i+2}, \hat{p}_{i+2})$$

where $p_{i+2}$ and $\hat{p}_{i+2}$ are probability distributions over the vocabulary of size $V$.

**What Cross Entropy computes:** $\text{CrossEntropy}(p, \hat{p}) = -\sum_{v=1}^{V} p(v) \log \hat{p}(v)$. Since $p_{i+2}$ is the target LLM's own output distribution for the next token (given the true feature), this term penalizes the draft model for producing features that lead to token distributions diverging from the target LLM's.

**Why this form over using ground-truth token labels:** The paper uses the target LLM's full distribution $p_{i+2}$ as the target, not the one-hot encoding of the actual next token $t_{i+2}$. This is a deliberate choice: the objective is to learn features that produce token distributions matching the target LLM's, not to predict a particular sampled token. Using the full distribution as the target incorporates information about the target LLM's uncertainty—tokens that the target LLM considers nearly equally probable should produce similar feature predictions. Additionally, this avoids overfitting to the particular sampling outcomes in the training data.

**Combined loss:**

$$\mathcal{L} = \mathcal{L}_{\text{reg}} + w_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}$$

where $w_{\text{cls}} = 0.1$ is a weighting coefficient.

**Why $w_{\text{cls}} = 0.1$:** The paper notes that "classification loss is an order of magnitude larger than the regression loss in numerical terms." The regression loss operates on feature vectors with element values that may be large in magnitude, while the classification loss is a cross-entropy over a 32,000-way distribution (which yields values typically in the range of 5–15 nats). Without the weighting coefficient, the classification loss would dominate the training signal, potentially causing the model to optimize for token-distribution matching at the expense of feature prediction accuracy. The factor 0.1 brings the two loss terms to comparable scales.

**Why both losses together?** The regression loss provides a direct signal for the draft model's primary intermediate task (predicting the correct feature). The classification loss provides an end-to-end signal that penalizes feature predictions that, while perhaps numerically close in L1/L2 sense, produce substantially wrong token distributions. The combination ensures that the draft model learns feature representations that are both numerically accurate and semantically meaningful (i.e., the features' projections through the LM Head yield the right token probabilities). This is important because small feature errors in certain dimensions can cause large changes in the LM Head's output (if those dimensions are heavily weighted in the LM Head projection), while large errors in other dimensions might be harmless.

**Noise augmentation.** To improve robustness to error accumulation during autoregressive drafting, the paper applies random noise to the target LLM's features during training:

$$f_j^{\text{noisy}} = f_j + \epsilon, \quad \epsilon \sim U(-0.1, 0.1)$$

where $U(-0.1, 0.1)$ is the uniform distribution over the interval $[-0.1, 0.1]$, applied element-wise to the feature vector.

**Why noise augmentation:** During the drafting phase, the draft model autoregressively predicts features—that is, after predicting $\hat{f}_{i+1}$, it uses this predicted feature (not the ground-truth feature from the target LLM) as input for predicting $\hat{f}_{i+2}$. Any error in $\hat{f}_{i+1}$ propagates to subsequent predictions. The draft model is trained on ground-truth features from the target LLM, so at training time it never sees feature inputs with errors. The noise augmentation simulates the error accumulation that occurs during inference, making the draft model robust to inexact feature inputs. The uniform noise distribution $U(-0.1, 0.1)$ is chosen as a simple way to inject errors of comparable magnitude to what the draft model might produce. The paper notes (Section 3.2) that this technique follows Jain et al. (2023), where similar noise augmentation improved fine-tuning robustness.

**Training hyperparameters.** The paper specifies (Section 4): learning rate of $3 \times 10^{-5}$, AdamW optimizer with $\beta_1 = 0.9$ and $\beta_2 = 0.95$, and gradient clipping at 0.5. Training is conducted for 1–2 days on 4× A100 (40G) GPUs for the 70B model, or on a single RTX 3090 node for 7B/13B/33B models. The trainable parameter counts scale with the target LLM's hidden dimension: 0.24B parameters for 7B models, 0.37B for 13B, 0.56B for 33B, 0.99B for 70B, and 0.28B for Mixtral 8x7B.

**Why these hyperparameters:** The learning rate $3 \times 10^{-5}$ is a standard choice for fine-tuning transformer models—low enough to avoid destabilizing the pre-trained decoder layer weights (if any are used for initialization) but high enough to converge in a reasonable number of epochs. AdamW is the standard optimizer for transformer training, decoupling weight decay from the adaptive learning rate. The gradient clipping at 0.5 prevents large gradient spikes from destabilizing training. The relatively short training duration (1–2 days) is possible because the model is small and the training data volume is modest (68,000 dialogues).

**Training data sensitivity (Section 4.3.3, Table 6).** The paper's ablation shows that EAGLE is not highly sensitive to the choice of training data. Using a fixed dataset (ShareGPT) versus generating training data from the target LLM autoregressively yields very similar results: 2.78× speedup vs. 2.88× speedup on LLaMA2-Chat 7B, with $\tau = 3.62$ vs. $\tau = 3.75$. The paper attributes this robustness to the fact that what the draft model learns is primarily the feature-level autoregressive dynamics of the target LLM—a structured mapping from (feature history, token history) to next feature. This mapping is largely determined by the target LLM's architecture and weights, and can be learned from any reasonably diverse text data that exercises the model's feature space. The target LLM doesn't need to have generated the training data itself; the features it produces on human-written text are sufficiently informative.

---

#### Tree-Structured Draft Generation

EAGLE generates not a linear chain of draft tokens but a **tree-structured draft**, where at each position the draft model produces multiple candidate tokens (each serving as the root of a sub-tree of further candidates). This increases the number of draft tokens that can be verified per forward pass of the target LLM, improving the potential speedup. The tree generation mechanism is described in Section 3.1, with the specific tree structure detailed in Appendix A.1 (Figure 9).

**How tree drafting works.** Given the current context, the draft model produces a predicted feature $\hat{f}$, and the LM Head produces a distribution $\hat{p}$ over the vocabulary. In greedy mode (temperature 0), the top $k$ tokens with the highest probabilities are selected as child nodes. In non-greedy mode (temperature > 0), $k$ tokens are sampled from $\hat{p}$. Each selected token forms a branch, and from each branch, the draft model continues autoregressively: it uses the predicted feature and the sampled token for that branch as input to predict the next feature, from which $k'$ tokens are sampled (where $k'$ may be smaller than $k$ for deeper positions in the tree). This process continues for a specified depth.

The tree is generated using multiple forward passes of the draft model, but these passes process multiple nodes in parallel through **tree attention**. At each step, the draft model processes the concatenated feature+token inputs for all active tree nodes simultaneously, with an attention mask that restricts each node to attend only to its ancestors in the tree (causal masking along each path, plus masking across unrelated branches). This enables generating many candidate tokens with relatively few draft-model forward passes.

**The specific tree structure used.** The paper uses a fixed tree structure for all experiments (Appendix A.1, Figure 9 left panel), which was "not rigorously optimized but rather based on intuition: branches of higher-probability tokens should be deeper and wider" (Section A.1). The root node branches into 4 children (top-4 tokens). These 4 children then branch with decreasing fan-out: 3 children each, then 2, then 1, creating a tree of depth 5 (or approximately 5 forward passes of the draft model) containing more than 10 total draft tokens. The total number of tokens in the draft tree exceeds the depth (number of forward passes) because at the root and early layers, multiple branches are explored in parallel.

**Why a tree rather than a chain?** A linear chain of length $\gamma$ produces exactly $\gamma$ draft tokens for the target LLM to verify. A tree with $m$ forward passes can produce $N_{\text{tokens}} > m$ tokens, because nodes at the same depth can branch in parallel. The verification phase processes all tokens in the tree in a single target-LLM forward pass, accepting a subset that forms a valid path through the tree. The average acceptance length $\tau$ (tokens accepted per verification pass) is higher for a tree than for a chain of the same forward-pass count, because the tree provides the verification algorithm with more options—if one branch is rejected early, tokens from another branch at the same depth can still be accepted.

The ablation in Section 4.3.1 (Table 5, Figure 7) quantifies this benefit. On Vicuna 7B, the average acceptance length increases from $\tau = 3.20$ (chain) to $\tau = 3.94$ (tree), a gain of +0.74, and the speedup ratio increases by approximately 0.3×–0.5×. The speedup improvement is smaller than the $\tau$ improvement because tree verification processes more tokens per forward pass, increasing the computational cost per target-LLM invocation (each verification forward pass processes all tokens in the tree, not just the accepted ones).

**Relationship to acceptance rate metrics (Section 4).** For chain drafts, the acceptance rate $\alpha$ is simply the ratio of accepted tokens to total drafted tokens. For tree drafts, this metric is less meaningful because multiple tokens are sampled at each position with only one ultimately accepted. The paper introduces **$n$-$\alpha$** as a tree-aware metric: the acceptance rate when the draft model has predicted a sequence of $n$ features with potentially $n$ inaccuracies. Specifically, $0$-$\alpha$ measures the acceptance rate for draft tokens whose entire upstream feature sequence was correct (no errors in any ancestor features), $1$-$\alpha$ measures acceptance when exactly one upstream feature was inaccurate, and so on. The gap between $0$-$\alpha$ and $1$-$\alpha$ (e.g., 0.79 vs. 0.74 for Vicuna 7B at temperature 0, Table 2) shows the impact of feature prediction errors on draft quality, while the relatively small decline from $1$-$\alpha$ to $4$-$\alpha$ indicates that error accumulation is not catastrophic—the draft model remains useful even after multiple rounds of imperfect feature predictions.

**The draft model's forward pass count.** Crucially, regardless of tree depth, the draft model always performs exactly the same number of forward passes per drafting phase as it would for a chain of the same depth. The tree structure increases the number of tokens processed per forward pass (since multiple branches are active) but does not increase the number of sequential forward passes. The cost of drafting is therefore proportional to tree depth (number of sequential autoregressive steps), not to the total number of draft tokens generated. This is the key to achieving high $\tau$ at low draft-model latency.

---

#### Verification Phase: Multi-Round Speculative Sampling on Trees

The verification phase ensures that, despite the draft model's potential inaccuracies and the tree structure, the final generated tokens are distributed exactly as if they had been produced by the target LLM's vanilla autoregressive decoding. EAGLE's verification algorithm is detailed in Appendix A.2 (Algorithm 1).

**Standard speculative sampling verification (chain).** For a linear chain of draft tokens, the verification algorithm (Leviathan et al., 2023) works as follows. The target LLM processes the full prefix plus the $\gamma$ draft tokens in a single forward pass, yielding true probabilities $p_{i+1}, p_{i+2}, \ldots, p_{i+\gamma+1}$ for each position. For each draft token $\hat{t}_{i+k}$ in sequence:

1. If $p_{i+k}(\hat{t}_{i+k}) \geq \hat{p}_{i+k}(\hat{t}_{i+k})$ (the target LLM assigns equal or higher probability to this token than the draft model did), accept the token.
2. If $p_{i+k}(\hat{t}_{i+k}) < \hat{p}_{i+k}(\hat{t}_{i+k})$, accept the token with probability $p_{i+k}(\hat{t}_{i+k}) / \hat{p}_{i+k}(\hat{t}_{i+k})$; otherwise reject.
3. On the first rejection, discard all subsequent draft tokens and resample a token from the adjusted distribution $\text{norm}(\max(0, p_{i+k} - \hat{p}_{i+k}))$.

This procedure is proven to produce tokens from the target LLM's exact distribution, because the acceptance probability is a Metropolis-Hastings step with the draft model as the proposal distribution.

**Multi-round sampling for tree structures (Algorithm 1).** EAGLE's verification adapts this to tree-structured drafts. The target LLM computes the true probability for every token in the tree in a single forward pass (with the tree's attention structure ensuring each token sees its valid prefix context). The algorithm then recursively applies the acceptance/rejection logic:

The algorithm's pseudocode (Appendix A.2) takes as input the target distribution $p$, and a sequence of draft tokens $\hat{t}_1, \hat{t}_2, \ldots, \hat{t}_k$ with their corresponding draft distributions $\hat{p}_1, \hat{p}_2, \ldots, \hat{p}_k$, where each $\hat{t}_i$ was sampled from $\hat{p}_i$. It iterates through $i = 1$ to $k$:

- Draw a random number $r \sim U(0, 1)$.
- If $r < p(\hat{t}_i) / \hat{p}_i(\hat{t}_i)$, **accept** $\hat{t}_i$ and return it immediately (no further tokens are considered—the path through the tree follows this accepted token).
- Otherwise ($r \geq p(\hat{t}_i) / \hat{p}_i(\hat{t}_i)$), **reject** $\hat{t}_i$, adjust the target distribution: $p \leftarrow \text{norm}(\max(0, p(\hat{t}_i) - \hat{p}_i(\hat{t}_i)))$, and continue to the next candidate token $i+1$.
- If all $k$ tokens are rejected, sample a token $\hat{t}$ from the final adjusted distribution $p$ and return $\hat{t}$.

**Why this procedure differs from chain verification.** In a chain-structured draft, when a token is rejected, all subsequent tokens are discarded and the algorithm terminates. In a tree-structured draft, multiple sibling tokens may exist at the same position (different branches from the same parent). When one sibling is rejected, the algorithm can try the next sibling at the same position, applying the adjusted distribution. This means the tree verification can accept tokens from branches that were not the draft model's single highest-probability path, increasing the effective acceptance length. The term "multi-round" reflects that the algorithm can loop through multiple candidate tokens at the same sequential position before deciding to resample from scratch.

The adjusted distribution $\text{norm}(\max(0, p - \hat{p}))$ after each rejection has a precise probabilistic interpretation. By subtracting the rejected draft probabilities from the target distribution and renormalizing, the algorithm accounts for the fact that the rejected tokens would have been sampled with probability $\hat{p}$ under the draft model. The resampling from this adjusted distribution ensures that the overall sampling probability of each token equals what it would have been under vanilla autoregressive decoding—the correction for rejection sampling.

**Why the verification preserves output distribution.** The proof in speculative sampling (Leviathan et al., 2023, Appendix A.1) does not depend on the draft being a chain. It relies on the fact that for any proposed token with draft probability $\hat{p}$ and target probability $p$, the acceptance probability $\min(1, p/\hat{p})$ combined with resampling from $\text{norm}(\max(0, p - \hat{p}))$ on rejection yields a final sampling distribution equal to $p$. The tree structure only changes which tokens are proposed—the mathematical guarantee holds per-node as long as each node's verification follows the same acceptance rule. The recursion in Algorithm 1 simply organizes multiple proposals at the same position into a sequence, applying the same guarantee at each step. The paper is explicit: "EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings" (Section 1).

**Recording accepted features for the next drafting round.** A practical detail: during verification, when tokens are accepted, the target LLM's forward pass naturally produces the corresponding features (second-to-top-layer hidden states) for those tokens. EAGLE records these accepted features alongside the accepted tokens, so that the next drafting phase has access to ground-truth features (not draft-model-predicted features) as input. This resets the error accumulation cycle: each new verification phase provides fresh, exact features from the target LLM, which the draft model uses to anchor its next round of feature predictions.

---

#### The Full Inference Loop

EAGLE's inference process alternates between drafting and verification in a loop, with the accepted features from verification feeding back as inputs to the next drafting phase. The complete cycle for generating a response:

**Phase 0: Initial prompt processing.** The target LLM processes the user prompt (all input tokens) in a standard forward pass, producing features $F_{1:N}$ for all $N$ prompt tokens. These features are saved.

**Phase 1: First drafting round.** Using the prompt features $F_{1:N}$ and the prompt tokens $T_{1:N}$, the draft model begins autoregressive feature prediction for the response:
- Position $N+1$: Input is $(F_{1:N}, T_{2:N+1})$ where $T_{N+1}$ is not known yet. For the very first draft token, $T_{N+1}$ is set to the target LLM's sampled first response token (from the prompt processing), or the draft model generates it from $F_N$.
- The draft model predicts $\hat{f}_{N+1}$, LM Head produces $\hat{p}_{N+2}$, sample $\hat{t}_{N+2}$.
- Continue for the desired tree depth (typically 5 forward passes), building the tree-structured draft.

**Phase 2: Verification.** The target LLM processes the entire prompt plus the draft tree in one forward pass, computing true probabilities for all nodes in the tree. Multi-round speculative sampling (Algorithm 1) determines which tokens are accepted. The accepted sequence—potentially including some tokens from the draft tree and possibly a resampled token if all proposals at some position are rejected—is appended to the output. The target LLM's features for the accepted tokens are saved.

**Phase 3: Subsequent drafting rounds.** The newly accepted tokens and their target-LLM-computed features are appended to the context. The next drafting round uses this extended context to generate a new tree of draft tokens, continuing the autoregressive process.

The loop continues until a stopping condition is met (end-of-sequence token generated, maximum length reached, etc.). Each verification round produces some number of accepted tokens $a \geq 1$ (at minimum, one token is always produced, even if all draft tokens are rejected, because of the resampling step). The average acceptance length $\tau$ measures the mean $a$ across all verification rounds.

**Why the loop resets feature error accumulation.** A critical property of this alternating design is that each verification phase provides fresh, exact features from the target LLM for all accepted tokens. This means the draft model always starts each drafting phase from ground-truth features—the only errors it encounters are in the features it predicts itself during that drafting phase. Error accumulation is limited to the depth of a single draft tree (typically 5 steps), not compounded across multiple drafting rounds indefinitely. This is why the paper observes (Table 2) that $1$-$\alpha$ through $4$-$\alpha$ show only a modest decline from $0$-$\alpha$—the draft model is robust to a few steps of imperfect feature inputs, and the loop design prevents errors from accumulating beyond the tree depth.

---

#### Computational Overhead and Parameter Efficiency

**Trainable parameters.** EAGLE's trainable parameters are solely those of the FC layer and the single decoder layer in the Autoregression Head. The FC layer has weight matrix $W_{\text{FC}} \in \mathbb{R}^{\text{hidden\_dim} \times 2\cdot\text{hidden\_dim}}$ and bias $b_{\text{FC}} \in \mathbb{R}^{\text{hidden\_dim}}$, for a total of $2 \cdot \text{hidden\_dim}^2 + \text{hidden\_dim}$ parameters. The decoder layer's parameter count depends on the specific transformer architecture (number of attention heads, feed-forward dimension) but is approximately $4 \cdot \text{hidden\_dim}^2$ for the self-attention components and $8 \cdot \text{hidden\_dim}^2$ for the feed-forward network (assuming standard FFN expansion factor of 4 and gated activation). The total trainable parameters scale as $O(\text{hidden\_dim}^2)$.

For LLaMA 7B (hidden_dim = 4096): 0.24B parameters.
For LLaMA 13B (hidden_dim = 5120): 0.37B parameters.
For LLaMA 33B (hidden_dim = 6656): 0.56B parameters.
For LLaMA 70B (hidden_dim = 8192): 0.99B parameters.
For Mixtral 8x7B (hidden_dim = 4096): 0.28B parameters.

**Inference overhead.** During drafting, the draft model must execute its forward passes. For a tree of depth $m$, this requires $m$ sequential forward passes of a single decoder layer. The target LLM, by contrast, requires 32–80 decoder layers per forward pass. The draft-model overhead per verification round is therefore approximately $m \times (1 / L)$ times the cost of a target-LLM forward pass, where $L$ is the number of layers in the target LLM. For a 70B model with $L = 80$ and a draft tree of depth $m = 5$, the draft-model overhead is approximately $5/80 \approx 6.25\%$ of a target-LLM forward pass per verification round. This is several orders of magnitude cheaper than using a separate 7B model (with $L = 32$ layers) as a draft model.

During verification, the target LLM processes all tokens in the draft tree in one forward pass. The computational cost of this forward pass is higher than a vanilla single-token forward pass (because more tokens are processed), but not proportionally to the number of draft tokens—much of the computation is in the early layers, which process the entire prefix plus draft tokens, and the cost of processing additional tokens in the later layers is dominated by memory access (since LLM inference is memory-bound). The net effect is that EAGLE produces more tokens per verification pass than vanilla decoding produces per forward pass, at only a modest increase in per-pass latency.

**Memory overhead.** EAGLE requires storing the draft model's parameters (0.24B–0.99B parameters, or 0.5–2 GB in FP16) alongside the target LLM's parameters. This increases the total GPU memory footprint compared to running only the target LLM. The paper acknowledges this in Section 4.4: under a 24 GB memory constraint (single RTX 3090), the maximum batch size decreases from 8 (vanilla) to 7 (EAGLE) for Vicuna 7B. For LLaMA2-Chat 70B on 4× A100 (40G) GPUs, the maximum batch size decreases from 5 to 4. This memory overhead is the trade-off for the latency improvement, and in throughput-constrained scenarios (where maximum batch size is the limiting factor), the throughput gain (approximately 2×, Table 7) is less than the latency speedup (approximately 3×).

---

#### Design Rationale Summary

**Why feature-level rather than token-level autoregression?** Features live in a continuous space with learned structure; tokens are discrete symbols in a large vocabulary. Predicting a continuous vector is a regression problem that benefits from smoothness and proximity in representation space; predicting a discrete token is a classification problem over 32,000+ categories. The draft model can learn feature dynamics more easily because semantically similar continuations produce similar features. The LM Head (borrowed from the target LLM) handles the feature-to-token mapping perfectly.

**Why shifted tokens rather than aligned tokens?** Aligned tokens ($t_j$ paired with $f_j$) provide semantic information but do not resolve the sampling ambiguity—knowing that "I" was at position $j$ tells you what feature $f_j$ should look like, but not what the *next* feature $f_{j+1}$ is (because the token $t_{j+1}$ that determines $f_{j+1}$ hasn't been seen). Shifted tokens ($t_{j+1}$ paired with feature $f_j$) provide the specific sampling outcome that disambiguates the feature transition. This is the paper's core innovation.

**Why trainable component is only one decoder layer + FC?** Keeping the draft model extremely lightweight minimizes the drafting latency overhead. Since LLM inference is memory-bound, adding a full small LLM as a draft model (as in standard speculative sampling) consumes memory bandwidth and compute that could otherwise be used for the target LLM. A single decoder layer adds minimal overhead while providing sufficient capacity to learn the feature autoregressive dynamics.

**Why tree attention rather than chain?** A tree generates more candidate tokens per forward pass than a chain of the same depth, increasing the number of tokens the target LLM can potentially accept per verification pass. The tree structure aligns with the speculative sampling philosophy—generate multiple plausible continuations and let the target model select the best one—while exploiting parallelism to amortize the draft model's per-pass cost across multiple branches.

**Why combined regression + classification loss?** The regression loss directly optimizes the intermediate task (predict the correct feature). The classification loss provides an end-to-end signal that penalizes feature errors that cause incorrect token distributions. The combination prevents the model from learning features that are numerically close to the target but semantically wrong. The weighting $w_{\text{cls}} = 0.1$ balances the magnitude disparity between the two losses.

**Why noise augmentation?** Error accumulation is an inherent challenge in any autoregressive prediction system—the draft model at inference time uses its own previous predictions as inputs, which contain errors. Noise augmentation during training simulates this condition, making the model robust to inexact feature inputs and preventing catastrophic error propagation across multiple draft steps.

**Why fixed training data rather than on-policy data?** Generating training data from the target LLM autoregressively is expensive (requires running the full LLM repeatedly). EAGLE's draft model learns the feature dynamics of the target LLM, which can be observed from the target LLM's features on any sufficiently diverse text. The ablation (Section 4.3.3) confirms that the performance difference is small, justifying the fixed-dataset approach for training cost reduction.

## 4. Key Insights and Innovations

### Innovation 1: Diagnosing Feature Uncertainty as THE Bottleneck in Lightweight Draft Models

The paper's most intellectually distinctive contribution is not any architectural choice but a **diagnostic insight**: it identifies a specific, previously unarticulated failure mode that explains why prior lightweight draft methods (Medusa, Lookahead) achieve limited accuracy, and shows that fixing this one problem unlocks the majority of the available speedup.

**What the field assumed before.** The dominant assumption in speculative sampling research was that draft quality was primarily limited by model capacity—a smaller draft model (Medusa's MLP heads, Lookahead's n-gram matching) simply lacked the parameters or representational power to match the target LLM's token distribution. The natural response was to either increase capacity (DistillSpec's knowledge distillation to improve the draft model) or accept the accuracy-capacity tradeoff as fundamental. The paper's ablation in Figure 8 reveals that this assumption was **wrong in a precise and actionable way**.

**The diagnostic move.** EAGLE decomposes the draft prediction problem into two orthogonal factors: (1) the **representation level** at which prediction occurs (tokens vs. features) and (2) whether the **sampling uncertainty** is resolved (do we know which token was actually sampled at the previous step?). The ablation tests all four combinations (token vs. feature × shifted vs. unshifted) on an identical draft model architecture, isolating each factor's contribution. The results on Vicuna 7B are striking:

- Token-level prediction without uncertainty resolution (standard autoregression over tokens): ~1.5× speedup
- Feature-level prediction without uncertainty resolution (feature-only, Medusa-like): ~1.9× speedup
- Feature-level prediction with partial uncertainty resolution (feature + aligned tokens, which provide semantic context but don't disambiguate the next feature): ~2.1× speedup
- Feature-level prediction with full uncertainty resolution (feature + shifted tokens, EAGLE): ~2.8× speedup

The critical finding is that **resolving sampling uncertainty accounts for a ~0.9× speedup increment** (from 1.9× to 2.8×), which is larger than the gain from switching from token-level to feature-level prediction (~0.4×, from 1.5× to 1.9×). This means the primary bottleneck in prior lightweight draft methods was not capacity—it was **informational**: the draft model lacked access to the sampling outcomes that determine which feature trajectory to follow. Medusa and Lookahead failed primarily because they operated under uncertainty, not because their MLP heads or n-gram models were too small.

**What makes this a fundamental contribution.** This diagnosis reframes the speculative sampling problem from "how do we build a faster/cheaper/more accurate draft model?" to "what information does the draft model need access to, and at what representation level should it operate?" It treats the draft model not as a smaller version of the target LLM but as a system that predicts the target LLM's internal state dynamics—and identifies that the dynamics have a specific structure (the next feature depends deterministically on the current feature + the sampling outcome) that can be exploited if the draft model has the right input information. This is a conceptual shift from capacity-scaling to information-design.

**Evidence.** Figure 8 (Section 4.3.2) provides the direct experimental decomposition. Figure 4 (Section 1 introduction) shows the convergence behavior—feature+shifted-token training stabilizes to higher speedup and accuracy than either feature-only or token-only, with the gap persisting across epochs rather than being a transient training artifact. Table 2's n-α metrics provide corroborating evidence: 0-α (acceptance rate when the upstream feature sequence has zero errors) is consistently and substantially higher than 1-α (one upstream error), confirming that feature prediction accuracy—which depends on having the correct sampling information—directly controls draft quality.

**Comparison to prior work.** No prior speculative sampling method analyzed the informational structure of the prediction problem. Standard speculative sampling (Leviathan et al., 2023) used a full smaller LLM as draft model, which implicitly resolves uncertainty because the smaller model performs its own autoregressive token sampling—but at the cost of running a complete transformer stack for each draft step. Medusa (Cai et al., 2023) predicted tokens from a single feature vector without addressing the intermediate sampling ambiguity, accepting the resulting accuracy loss as the price of lightweight drafting. Lookahead (Fu et al., 2023) used Jacobi iteration over n-grams, which is a heuristic that partially captures uncertainty via iterative refinement but lacks the explicit shifted-token conditioning. EAGLE's diagnostic contribution is isolating uncertainty as the specific factor that prior methods failed to address, and demonstrating that addressing it with a minimal information input (one shifted token per position) yields gains comparable to or exceeding those from much more complex architectural changes.

---

### Innovation 2: Establishing Feature-Level Autoregression as the Right Abstraction Level for Drafting

EAGLE makes a specific architectural claim that goes beyond the uncertainty diagnosis: **the second-to-top-layer feature space of the target LLM is the correct level of abstraction at which to perform autoregressive prediction for drafting purposes**. This is not obvious a priori—one could imagine predicting at the token level (standard autoregression), at the embedding level, at some intermediate transformer layer, or at the logit level. The paper provides both theoretical argument and empirical evidence for why the second-to-top-layer features are uniquely suited to the drafting task.

**What makes this non-obvious.** In a standard LLM, the mapping from features to tokens is linear (the LM Head is typically a single matrix multiplication followed by softmax). This means features and token logits are essentially equivalent up to a linear transformation—predicting one should be as hard as predicting the other, since the mapping is invertible (for a full-rank LM Head matrix). Why, then, does feature-level prediction outperform token-level prediction by ~0.4× speedup in the ablation (Figure 8)?

The answer lies in the **structure of the loss landscape and the inductive bias of the draft model architecture**. Token-level prediction is a classification problem over a vocabulary of 32,000+ categories. The draft model must learn to assign high probability to exactly the correct token(s) among this large discrete set. Small errors in the predicted representation can cause the argmax to jump between semantically unrelated tokens, producing completely wrong drafts. Feature-level prediction is a regression problem in a continuous space where the draft model's output is a vector in R^hidden_dim. The Smooth L1 regression loss provides a smoother training signal that encourages the predicted feature to be numerically close to the ground-truth feature, and two features that are numerically close are also functionally close (their LM Head projections produce similar token distributions).

In other words, the feature space comes with a **natural metric** (Euclidean distance in R^hidden_dim) that correlates with downstream task performance, because the target LLM's training has already organized the feature space so that similar continuations map to nearby features. The token space has no such metric—two tokens with adjacent indices have no semantic relationship, and the cross-entropy loss provides a cruder training signal for the draft model.

**Why this matters beyond the numbers.** The paper is making a claim about **where to plug in** when building auxiliary models that interact with frozen LLMs. For any task that requires predicting or manipulating an LLM's future behavior—not just speculative sampling but also interpretability, controlled generation, or model editing—the second-to-top-layer feature space may be a more tractable interface than the token or embedding spaces. The paper demonstrates this empirically for drafting, but the principle generalizes: the feature space is where the LLM's "understanding" lives, compressed into a representation that is simultaneously information-rich (it encodes everything needed to predict the next token) and structured (nearby features correspond to similar predictions).

**Evidence for the abstraction choice.** Figure 4 shows the convergence curves: feature-level and feature+shifted-token models consistently outperform the token-level model across all epochs, not just at convergence. This indicates the advantage is not a training artifact but a fundamental property of the prediction task. The acceptance rate metrics in Table 2 provide additional support: even the feature+unshifted-token variant (which resolves some but not all uncertainty) outperforms token-only prediction, suggesting the feature-level abstraction provides benefits independent of uncertainty resolution.

**Relationship to Medusa's design.** Medusa also uses second-to-top-layer features as input—its heads predict tokens from $f_j$ directly. But Medusa treats features as an input representation, not as a prediction target. It uses features to predict tokens in one shot, without autoregressing through feature space. EAGLE's innovation is recognizing that features should be both the input representation AND the autoregressive target—that the dynamics of features through time are more learnable than the dynamics of tokens. This is what distinguishes "predicting from features" (Medusa) from "predicting features autoregressively" (EAGLE), and the speedup difference (~1.5×-1.6× for Medusa vs. ~2.8×-3.0× for EAGLE on comparable models, per Figure 1) validates that autoregressing through feature space is the key mechanism, not just using features as inputs.

---

### Innovation 3: A Unifying Framework for Lossless Speculative Sampling Without Architectural Compromise

EAGLE synthesizes three properties that prior speculative sampling methods achieved only in partial or compromised forms: **lossless distribution preservation**, **applicability to both greedy and non-greedy decoding**, and **no dependence on a separate pre-trained draft model of the right size**. Any individual property had been demonstrated before, but the combination—and the specific way EAGLE achieves it—represents a genuine advance in the speculative sampling design space.

**The tradeoff landscape before EAGLE.** Standard speculative sampling (Leviathan et al., 2023; Chen et al., 2023a) provides lossless preservation for both greedy and non-greedy settings, but requires a separate smaller LLM as draft model. This fails for the smallest model in a series (no draft model exists) and underperforms even for larger models when draft-model overhead is high (the paper reports only 1.12× speedup for 33B and 1.88× for 70B with a 7B draft model). DistillSpec (Zhou et al., 2023) improves the draft model through distillation but doesn't eliminate the separate-model requirement and "the bottleneck for speculative sampling performance lies in the high overhead of the draft model" (Section 4.1). Self-Speculative Decoding (Zhang et al., 2023) skips layers of the target LLM to create a draft model, which is architecturally elegant but still incurs the cost of running a subset of the target's transformer layers.

Medusa (Cai et al., 2023) eliminates the separate draft model entirely by attaching lightweight heads to the target LLM, achieving low overhead. But Medusa sacrifices the distribution preservation guarantee for non-greedy settings—the paper explicitly notes (Figure 2 caption) that "the non-greedy generation of Medusa does not guarantee lossless performance." This is because Medusa's independent per-token prediction from a single feature vector does not produce a valid autoregressive proposal distribution over multi-token sequences; the speculative sampling acceptance/rejection guarantee breaks when the draft model does not define a proper sequential sampling process.

Lookahead (Fu et al., 2023) similarly eliminates the separate draft model but is "confined to greedy decoding" (Figure 2 caption). Its Jacobi iteration and n-gram matching produce candidate sequences that can be verified for greedy decoding but do not define a proper probability distribution for non-greedy settings.

**How EAGLE resolves the tradeoff.** EAGLE achieves all three properties simultaneously through a specific combination of design choices that each address one aspect of the tradeoff:

- **Lossless preservation** is achieved by using the standard speculative sampling verification procedure (Algorithm 1) unchanged. Because EAGLE's draft model produces tokens through a proper autoregressive process (predict feature → LM Head → sample token → feed back as input), it defines a valid sequential proposal distribution $\hat{p}(\hat{t}_{i+1} | \text{context}, \hat{t}_{1:i})$ for multi-token drafts. The acceptance/rejection guarantee from Leviathan et al. (2023) applies directly. This is in contrast to Medusa, whose independent heads do not produce properly conditioned sequential probabilities, meaning the acceptance probability $\min(1, p/\hat{p})$ is not well-defined for multi-token spans.

- **Greedy and non-greedy applicability** follows from the same property. Because EAGLE's draft model defines a proper autoregressive distribution, it can sample stochastically (temperature > 0) or greedily (temperature = 0), and the verification procedure handles both cases identically. Prior lightweight methods either restricted to greedy (Lookahead) or used heuristic truncation in non-greedy mode that breaks the distribution guarantee (Medusa's threshold-based acceptance).

- **No separate draft model** is achieved by making the draft model a lightweight plug-in (one decoder layer + FC layer) that reuses the target LLM's Embedding layer and LM Head. This eliminates the need for a pre-existing smaller model from the same series, solving the "no draft model for 7B target" problem. It also eliminates the draft-model overhead bottleneck that limited standard speculative sampling's speedup on 13B and 33B models—the single decoder layer is orders of magnitude cheaper than even the smallest complete LLM.

**Why this synthesis matters.** The three properties address three distinct deployment constraints. Lossless preservation matters for applications where output quality is paramount and any distribution shift is unacceptable (e.g., API services that guarantee identical behavior to the base model). Non-greedy support matters for creative generation, dialogue diversity, and any application where users set temperature > 0. Draft-model independence matters for model series where no suitable smaller variant exists, and for reducing total GPU memory footprint (no second full model to load). Prior to EAGLE, a practitioner had to choose which property to sacrifice. EAGLE shows that the sacrifice is unnecessary—the properties are jointly achievable through careful design of the prediction target (features), input conditioning (shifted tokens), and model architecture (single decoder layer reusing target LLM components).

**Evidence for the synthesis claim.** Figure 1 validates the speedup across model sizes without separate draft models (2.90× on Vicuna 7B, where standard speculative sampling is marked N/A because no draft model exists). Figure 2 validates the non-greedy speedup (2.13×–2.68× across models at temperature=1), where Lookahead is inapplicable and Medusa's lossless guarantee doesn't hold. The paper's description of the verification phase (Section 3.3, Appendix A.2) explicitly invokes the standard speculative sampling proof as the theoretical basis for distribution preservation.

**Is this a fundamental advance or an engineering synthesis?** The synthesis is fundamentally enabled by the two conceptual innovations above (diagnosing uncertainty, choosing feature-level autoregression). Without the shifted-token conditioning to resolve uncertainty, a single-decoder-layer draft model would be too inaccurate to compete with standard speculative sampling (the feature-only ablation achieves only ~1.9× speedup, which is better than Medusa's ~1.5× but not transformative). Without feature-level autoregression, the draft model would need to predict tokens directly, requiring a larger architecture or more training data to achieve competitive accuracy. So the synthesis is not an independent innovation but rather a **consequence** of Innovations 1 and 2—the design choices that enable the synthesis are precisely the ones that resolve uncertainty and operate at the feature level. The paper's contribution is demonstrating that these choices jointly solve the three-way tradeoff, which was not obvious ex ante (since Medusa and Lookahead accepted compromises on one or more properties in exchange for lightweight drafting).

---

### Innovation 4: Demonstration That Draft-Model Training Can Be Decoupled From Target-LLM Data Generation

EAGLE's training methodology contains a finding with practical significance beyond the specific architecture: **the draft model does not need to be trained on data generated by the target LLM autoregressively; a fixed, off-the-shelf dataset works nearly as well**. This finding challenges the natural assumption in speculative sampling that the draft model should be trained to mimic the target LLM's behavior on the target LLM's own outputs—what we might call "on-policy" training.

**The natural expectation.** Standard speculative sampling uses a smaller LLM from the same series as the draft model. That smaller LLM was pre-trained independently (potentially on different data) but shares the same architecture and training paradigm. When building a custom draft model, the intuitive approach would be to train it to predict the target LLM's outputs—feed prompts through the target LLM, collect its generated responses, and train the draft model on those (token, feature) pairs. This is "on-policy" in the sense that the training data matches the distribution of sequences the target LLM produces at inference time. The expectation is that on-policy training would substantially outperform off-policy training (using a fixed dataset not generated by the target LLM), because the draft model needs to predict the target LLM's features, and those features depend on the target LLM's weights, not on arbitrary text.

**The surprising result.** Table 6 (Section 4.3.3) shows that training on the ShareGPT dataset (fixed, off-policy) yields 2.78× speedup and τ = 3.62 on LLaMA2-Chat 7B, while training on data generated by the target LLM (same questions from ShareGPT, but answers generated by LLaMA2-Chat 7B) yields 2.88× speedup and τ = 3.75. The difference is small—only 0.10× speedup and 0.13 in average acceptance length. This means the draft model achieves ~97% of its optimal speedup using completely off-policy training data.

**Why this is not obvious.** The target LLM's features $f_i$ depend on both the input text and the model's specific parameters. Different LLMs produce different feature sequences for the same text, because their internal representations learned during pre-training differ. A draft model trained on features from one distribution might not generalize to features from a different distribution. The natural concern is that features extracted from human-written text (ShareGPT) would differ systematically from features the target LLM produces when generating text autoregressively, and that this distribution shift would degrade draft accuracy.

The result suggests that the **feature-level autoregressive dynamics learned by EAGLE's draft model are largely invariant to the source of the text**, as long as the text is sufficiently diverse and the target LLM is the one computing the features. Whether the text was written by humans or generated by the target LLM, the mapping $(F_{1:i}, T_{2:i+1}) \to f_{i+1}$ is determined by the target LLM's architecture and weights, and appears to be stable across text distributions. This is consistent with the interpretation that the draft model learns the target LLM's internal dynamics—how features evolve from one token to the next—rather than learning a distribution over text content.

**Practical significance.** This finding dramatically reduces the cost of training EAGLE. Generating training data from the target LLM autoregressively is expensive: for 68,000 dialogues, each requiring potentially hundreds of tokens of generated response, the target LLM would need to run millions of forward passes. With the fixed-dataset approach, the target LLM runs only once per training example (to extract features), and it can process all training data in a single pass (no generation, just feature extraction). This is what enables the paper's claim of "1–2 days on 4× A100 GPUs" for training EAGLE on 70B models. If on-policy data generation were required, training time and cost would multiply by the average response length (easily 10–50×).

**Generalizability of the finding.** The paper only tests this on LLaMA2-Chat 7B, and only with the ShareGPT dataset. It is possible that for target LLMs with very different training distributions or architectures, the off-policy approach would perform worse. The paper also does not test how the gap scales with training data volume—perhaps the on-policy advantage would grow if less training data were available, since on-policy data might be more "efficient" per example. But for the specific setting tested (68,000 dialogues, LLaMA2-Chat target, ShareGPT data), the finding is clear and practically important.

**Comparison to prior work.** Standard speculative sampling requires no draft-model training at all (it uses a pre-existing smaller LLM), so this finding is not directly comparable. DistillSpec trains draft models using distillation, which typically requires target-LLM-generated data to define the distillation targets—the finding that off-policy data nearly matches on-policy data for feature prediction is novel to EAGLE's training paradigm. Medusa trains its heads using a fixed dataset (the paper does not specify whether it's target-LLM-generated), but Medusa's heads are MLPs that predict tokens directly from features, not autoregressive feature predictors—the training-data sensitivity may differ. EAGLE's finding is specific to its feature-level autoregressive approach and represents a practical advantage that reduces the barrier to adoption.

**Is this a fundamental discovery or a practical convenience?** The finding is primarily practically significant rather than theoretically deep, but it reveals something about the structure of the problem: the feature-level autoregressive mapping is **a property of the model, not of the data distribution**. This suggests that future work on model introspection, feature-level control, or interpretability might also benefit from training on off-policy data, since the target model's internal dynamics are stable across text sources. If this generalizes, it reduces the cost of any technique that requires learning to predict or manipulate LLM features.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on four datasets spanning different task types: **MT-bench** (Zheng et al., 2023), a multi-turn dialogue benchmark with 80 questions across 8 categories designed to simulate real-world chatbot interactions; **HumanEval** (Chen et al., 2021), a code generation benchmark with 164 programming problems; **GSM8K** (Cobbe et al., 2021), a mathematical reasoning dataset with grade-school math word problems; and **Alpaca** (Taori et al., 2023), an instruction-following dataset. The primary evaluation is on MT-bench because it "has been employed by the current state-of-the-art, including Lookahead and Medusa, to demonstrate their speedup ratios," facilitating direct comparison (Section 4). All experiments use the standard evaluation splits—MT-bench's 80 multi-turn dialogues, HumanEval's 164 problems, GSM8K's test set, and Alpaca's instruction set—with no custom splitting or held-out validation described.

- **Base model(s).** EAGLE is evaluated on all models from the **Vicuna** (7B, 13B, 33B) and **LLaMA2-Chat** (7B, 13B, 70B) series (Section 4). These span the common size range of contemporary open-source LLMs and include both the smallest (7B) and largest (70B) variants, directly testing the claim that EAGLE works where standard speculative sampling fails (i.e., on the 7B models that lack suitable smaller draft models). Additionally, EAGLE is tested on **Mixtral 8x7B Instruct**, a mixture-of-experts model, to probe the method's applicability to non-dense architectures. The models were chosen to be "representative of the capabilities of many contemporary LLMs" (Section 1) and to enable comparison with prior work that reported results on the same model families (Lookahead, Medusa).

- **Metrics.** The paper uses three primary metrics (Section 4):
  - **Walltime speedup ratio:** The actual measured inference latency of vanilla autoregressive decoding divided by the inference latency of EAGLE, computed end-to-end on the same hardware. This is the paper's headline metric and captures real-world acceleration including all overhead costs (draft-model forward passes, verification passes, tree attention, acceptance/rejection logic).
  - **Average acceptance length τ:** The mean number of tokens accepted per forward pass of the target LLM. This measures how effectively EAGLE reduces the target model's invocation frequency—vanilla decoding has τ = 1 (one token per pass), while EAGLE's τ ranges from 3.2 to 4.5 depending on the model and task.
  - **Acceptance rate α (n-α):** For chain-structured drafts (without tree attention, used in ablation), the fraction of draft tokens accepted by the verification procedure. For tree-structured drafts, the paper uses n-α—the acceptance rate conditioned on having n inaccurate features in the upstream sequence. 0-α (all upstream features correct) represents the draft model's accuracy under ideal conditions; 1-α through 4-α measure robustness to error accumulation. The paper explicitly notes that "evaluating the quality of EAGLE's generated results is both unnecessary and meaningless" because the speculative sampling verification guarantees lossless distribution preservation—the generated text is identical in distribution to vanilla decoding by construction.

- **Baselines.** The paper compares against three categories of methods:
  - **Standard speculative sampling** (Leviathan et al., 2023; Chen et al., 2023a): Uses the smallest available model from the same series as a draft model. For Vicuna and LLaMA2-Chat, the 7B variant serves as the draft model for 13B, 33B, and 70B targets. For the 7B target, there is no suitable smaller model (marked as "N/A" in Figure 1). The paper sweeps draft lengths from 2 to 10 and reports the best speedup obtained. **DistillSpec** (Zhou et al., 2023) is also tested, applying knowledge distillation to the draft model with the same training data as EAGLE and the FKL divergence function from the DistillSpec paper.
  - **Medusa** (Cai et al., 2023): Attaches multiple MLP heads to the target LLM's second-to-top-layer features to independently predict future tokens. Speedup numbers are taken directly from the Medusa technical report for Vicuna and LLaMA2-Chat models, with the paper noting that "the non-greedy generation of Medusa does not guarantee lossless performance" (Figure 2 caption).
  - **Lookahead** (Fu et al., 2023): Uses n-gram matching and Jacobi iteration to generate drafts, restricted to greedy decoding. Speedup numbers are copied from the Lookahead technical report. Lookahead is not compared in non-greedy settings because it "is confined to greedy decoding" (Figure 2 caption).
  - **Vanilla autoregressive decoding:** The standard baseline of generating one token per forward pass of the target LLM, representing a speedup ratio of 1.00×.

- **Generation budget / compute accounting.** EAGLE's speedup is measured as walltime latency reduction, not a token-generation budget. The paper does not constrain or allocate a generation budget—instead, it runs both vanilla decoding and EAGLE on the same prompts and measures the end-to-end walltime ratio. This accounts for ALL costs: draft-model forward passes (5 per drafting phase for the tree structure shown in Figure 9), target-LLM verification passes (1 per drafting round), tree attention overhead, and the acceptance/rejection computation. Comparison to other methods is based on reported speedup ratios from their respective papers, measured on the same MT-bench benchmark. The paper notes that "the majority of our experiments adopted [batch size 1]" (Section 4), with batch size > 1 explored in Section 4.4.

- **Cross-validation / statistical protocol.** The paper does not report cross-validation, statistical significance tests, confidence intervals, or multiple random seeds. Speedup ratios are reported as point estimates from single evaluation runs on the test sets. This is typical for speculative sampling papers (Leviathan et al., 2023; Cai et al., 2023; Fu et al., 2023), which report deterministic speedup measurements rather than statistical estimates, but it means there is no quantification of variance due to hardware conditions, sampling randomness (for temperature > 0), or prompt ordering effects.

---

### Main Quantitative Results

#### Speedup on MT-bench (Multi-Turn Dialogue): Greedy Decoding

The headline result appears in Figure 1 and Table 2 (Section 4.1). Across all Vicuna and LLaMA2-Chat models at temperature=0, EAGLE achieves speedup ratios of **2.78×–3.07×** relative to vanilla autoregressive decoding.

For LLaMA2-Chat 70B specifically, the paper reports a speedup ratio of **3.01×** (Figure 1), with average acceptance length τ = 3.81 and 0-α = 0.75 (Table 2). This means the 70B model produces approximately 3.81 tokens per verification forward pass, compared to 1.0 for vanilla decoding, directly accounting for the ~3× speedup.

For Vicuna 7B, where standard speculative sampling is marked "N/A" because no suitable draft model exists, EAGLE achieves **2.90×** speedup (Figure 1), demonstrating that the method solves the "no draft model for the smallest target" problem. The acceptance metrics for Vicuna 7B are τ = 3.94 and 0-α = 0.79 (Table 2).

The speedup ratios vary modestly across model sizes (2.78×–3.07× for LLaMA2-Chat, 2.90×–2.95× for Vicuna), indicating that EAGLE's effectiveness is relatively stable across the 7B-to-70B range. The paper attributes the small differences to the "complexities of accelerating MoE models" in the case of Mixtral 8x7B, which achieves only 1.50× speedup (Table 3).

**Comparison to standard speculative sampling.** EAGLE dramatically outperforms standard speculative sampling on all model sizes. For LLaMA2-Chat 70B, EAGLE achieves 3.01× vs. speculative sampling's 2.13× (with DistillSpec at 1.88×). For LLaMA2-Chat 13B, EAGLE achieves 3.03× vs. speculative sampling's 1.51×. For Vicuna 33B, EAGLE achieves 2.95× vs. speculative sampling's 1.00× (no speedup) vs. DistillSpec's 1.27×. The paper explicitly notes that for the 13B model, standard speculative sampling with a 7B draft model "did not achieve acceleration" (Figure 1, "N/A" for speculative sampling on 13B in some configurations), and for 33B/70B, the speedup ratios were modest (1.12× and 1.88×, respectively, as stated in Section 4.1).

**Comparison to Medusa and Lookahead.** EAGLE's 3.03× on LLaMA2-Chat 13B compares to Medusa's 1.64× and Lookahead's 1.00×—a 1.85× speedup over Medusa and 3.03× over Lookahead. Across all models, the paper states that EAGLE is "1.70x-2.08x" faster than Lookahead and "1.47x-1.60x" faster than Medusa (Section 4.1, Abstract). These comparison numbers are computed as ratios of the reported speedup ratios (e.g., 3.03× / 1.64× ≈ 1.85×, falling within the stated 1.47×–1.60× range which may represent slightly different model-task combinations).

**Why EAGLE outperforms speculative sampling.** The paper attributes standard speculative sampling's poor performance to the high overhead of the 7B draft model: "Employing a 7B model as the draft model for a 13B model results in slow speeds due to the high overhead of the 7B model, rendering it less efficient than vanilla autoregressive decoding" (Figure 1 caption). For DistillSpec, the paper observes that "while distillation slightly improved the speedup ratio, the limited enhancement is because distillation aims to increase the draft model's acceptance rate, while the bottleneck for speculative sampling performance lies in the high overhead of the draft model" (Section 4.1). EAGLE's draft model—a single decoder layer—eliminates this bottleneck.

#### Speedup on MT-bench: Non-Greedy Decoding (Temperature=1)

Figure 2 and Table 2 report results for temperature=1 generation. EAGLE achieves speedup ratios of **2.13×–3.07×** across Vicuna and LLaMA2-Chat models (Figure 2). Specifically:

- LLaMA2-Chat 70B: **2.67×** (vs. 3.01× at temperature=0)
- LLaMA2-Chat 13B: **2.68×** (vs. 3.03× at temperature=0)
- Vicuna 7B: **2.13×** (vs. 2.90× at temperature=0)
- Vicuna 13B: **2.32×** (vs. 3.07× at temperature=0)

The speedup ratios at temperature=1 are consistently lower than at temperature=0, which the paper attributes to the inherent difficulty of stochastic generation: at temperature>0, the draft model must match not just the most likely token but the entire sampling distribution, and stochastic sampling introduces additional randomness that makes draft prediction harder. The average acceptance length τ also decreases: for LLaMA2-Chat 70B, τ drops from 3.81 (temperature=0) to 3.46 (temperature=1) in Table 2; 0-α drops from 0.75 to 0.73.

**Comparison to baselines at temperature=1.** Lookahead is excluded from this comparison because it "is confined to greedy decoding." Medusa is excluded because "the non-greedy generation of Medusa does not guarantee lossless performance" (Figure 2 caption). Standard speculative sampling and DistillSpec are still applicable and achieve speedup ratios shown in Figure 2—for LLaMA2-Chat 70B, speculative sampling achieves 2.06× vs. EAGLE's 2.67×, and DistillSpec achieves 1.84×. The gap between EAGLE and speculative sampling narrows slightly at temperature=1 (2.67× vs. 2.06× = 1.30× advantage, compared to 3.01× vs. 2.13× = 1.41× at temperature=0), but EAGLE remains substantially faster.

**Non-greedy theoretical guarantee.** The paper emphasizes that EAGLE preserves the output distribution for non-greedy settings: "EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings" (Section 1). This is a key differentiator from Medusa, which does not guarantee lossless performance for non-greedy generation.

#### Speedup Across Different Tasks: HumanEval, GSM8K, Alpaca

Table 1 reports speedup ratios for code generation (HumanEval), mathematical reasoning (GSM8K), and instruction following (Alpaca) at both temperature=0 and temperature=1. The headline finding is that EAGLE achieves the **highest speedups on code generation (HumanEval)**.

At temperature=0 on HumanEval:
- LLaMA2-Chat 70B: **3.52×** (τ = 4.42)
- LLaMA2-Chat 13B: **3.76×** (τ = 4.52)
- Vicuna 33B: **3.67×** (τ = 4.28)
- Vicuna 7B: **3.33×** (τ = 4.29)

The paper attributes the superior performance on HumanEval to "the prevalence of fixed templates in code, making it easier to generate drafts for these templates" (Section 4.1). Code often contains repetitive syntactic structures (function signatures, loop constructs, variable declarations) that create predictable feature sequences, making the draft model's autoregressive feature prediction more accurate. This is reflected in the higher τ values for HumanEval (4.24–4.52) compared to MT-bench (3.62–3.98) and Alpaca (3.61–3.86) at temperature=0 (Tables 1 and 2).

At temperature=0 on GSM8K (mathematical reasoning):
- LLaMA2-Chat 70B: **3.03×** (τ = 3.93)
- Vicuna 7B: **3.01×** (τ = 4.00)
- Performance is slightly lower than HumanEval but comparable to MT-bench, suggesting that mathematical reasoning—despite its structured step-by-step format—is somewhat less predictable than code templates.

At temperature=0 on Alpaca (instruction following):
- LLaMA2-Chat 70B: **2.97×** (τ = 3.77)
- Vicuna 7B: **2.79×** (τ = 3.86)
- The lowest speedups across tasks, which the paper does not explicitly explain but which is consistent with Alpaca involving diverse, open-ended instruction-following responses that may be harder for the draft model to predict autoregressively.

At temperature=1, the speedup ratios for all tasks are lower than at temperature=0, following the same pattern observed on MT-bench. For HumanEval, LLaMA2-Chat 70B achieves 2.92× at temperature=1 vs. 3.52× at temperature=0; for GSM8K, 2.74× vs. 3.03×; for Alpaca, 2.65× vs. 2.97×. The relative ranking of tasks is preserved across temperatures.

#### Acceptance Rate Analysis: Understanding Why EAGLE Works

Table 2 provides the detailed n-α breakdown for all models on MT-bench. Three patterns are notable:

**Pattern 1: High 0-α indicates strong draft accuracy under ideal conditions.** For Vicuna 7B at temperature=0, 0-α = 0.79—meaning that when the draft model receives a feature sequence with zero prediction errors, its draft tokens are accepted 79% of the time by the target LLM's verification. For LLaMA2-Chat 13B, 0-α = 0.77. These numbers substantially exceed Medusa's reported accuracy of ~0.6 (Section 1) and explain why EAGLE can achieve higher τ despite similar draft-model overhead.

**Pattern 2: The gap between 0-α and 1-α quantifies the impact of feature prediction errors.** For Vicuna 7B at temperature=0, 0-α = 0.79 and 1-α = 0.74—a drop of 0.05. This means that having one erroneous feature in the upstream sequence reduces draft token acceptance by approximately 6.3% (relative). The paper highlights this gap: "the acceptance rate for completely accurate feature sequences, 0-α, significantly exceeds that for sequences with a single erroneous feature, 1-α, indicating the impact of feature errors on draft model performance" (Section 4.1).

**Pattern 3: The small decline from 1-α to 4-α indicates robustness to error accumulation.** For Vicuna 7B at temperature=0, the acceptance rates are 1-α = 0.74, 2-α = 0.72, 3-α = 0.73, 4-α = 0.67. The decline from 1-α to 3-α is minimal (0.74 → 0.73), and only at 4-α does a more significant drop occur (0.73 → 0.67). The paper interprets this as "EAGLE's robustness to feature errors and its adept handling of error accumulation" (Section 4.1). This robustness is attributed to two design choices: (1) noise augmentation during training, which exposes the draft model to inexact feature inputs, and (2) the alternating draft-verify-draft loop design, which resets feature errors after each verification round by providing fresh ground-truth features from the target LLM.

#### Mixtral 8x7B Results: MoE Model Performance

Table 3 reports results for Mixtral 8x7B Instruct-v0.1 at temperature=0 on MT-bench. EAGLE achieves a **1.50×** speedup with τ = 3.25 and 0-α = 0.67. This is substantially lower than the 2.78×–3.07× achieved on dense LLaMA2-Chat models, which the paper attributes to the complexity of accelerating mixture-of-experts models through speculative sampling.

The bottleneck is specific to MoE architectures: "In vanilla autoregressive decoding, processing a single token in an MoE model necessitates reading the weights of only two experts. However, during the verification phase of speculative sampling, processing multiple tokens simultaneously may necessitate accessing the weights of more than two experts, in contrast to dense models where all weights are always read in every forward pass" (Section 4.1, paraphrased from Table 3 discussion). In a dense decoder-only transformer, every forward pass reads all parameters regardless of sequence length—so processing a tree of draft tokens in one pass incurs no additional parameter-loading cost beyond what a single-token pass would incur. In an MoE model, each token activates only a subset of experts (typically top-2), but processing multiple tokens simultaneously may activate additional experts for different tokens, increasing the total parameter-loading cost per verification pass. This additional overhead reduces the net speedup from speculative sampling.

The acceptance rate (0-α = 0.67) and average acceptance length (τ = 3.25) are also lower than for dense models, but not dramatically so—the τ of 3.25 is comparable to LLaMA2-Chat 7B's 3.62. This suggests that EAGLE's draft-model accuracy on Mixtral is reasonable, but the MoE-specific verification overhead consumes much of the potential speedup gain.

#### Throughput Analysis: EAGLE at Batch Size > 1

Section 4.4 and Table 7 extend the evaluation to throughput (tokens per second) for batch sizes greater than 1, addressing the fact that inference is memory-bound and GPU computational resources become scarcer at larger batch sizes. The key findings are:

**Speedup diminishes with batch size.** For Vicuna 7B at temperature=0 on MT-bench, the speedup ratios are: 2.90× (bs=1), 2.87× (bs=2), 2.65× (bs=3), 2.76× (bs=4). For LLaMA2-Chat 70B: 3.01× (bs=1), 2.81× (bs=2), 2.50× (bs=3), 2.40× (bs=4). The paper explains: "As the batch size increases, the available computational capacity of the GPU decreases, leading to a reduction in the acceleration effect" (Section 4.4). With larger batch sizes, the GPU is more fully utilized by vanilla decoding, leaving less spare capacity for EAGLE's draft model to exploit.

**Anomalous increase at bs=4 for Vicuna 7B.** The speedup ratio increases from 2.65× (bs=3) to 2.76× (bs=4) for Vicuna 7B. The paper attributes this to the specific interaction between EAGLE's multi-token verification and GPU utilization: "during the verification phase of EAGLE, the target LLM processes multiple tokens in a single forward pass, and the processing at bs=4 is faster than at bs=3. In contrast, with vanilla autoregressive decoding where the target LLM processes one token per forward pass, the speeds at bs=3 and bs=4 are nearly identical" (Section 4.4). This suggests that the verification phase's parallelism interacts with batch-level parallelism in ways that can create non-monotonic scaling.

**Throughput under fixed memory constraints.** Under a 24 GB memory constraint (single RTX 3090), the maximum batch size for Vicuna 7B is 8 for vanilla decoding and 7 for EAGLE—the draft model's 0.24B parameters consume additional GPU memory. However, EAGLE achieves 1.97× higher throughput at its maximum batch size (bs=7) compared to vanilla decoding at its maximum batch size (bs=8). For LLaMA2-Chat 70B under a 160 GB constraint (4× A100 40G), the maximum batch sizes are 5 (vanilla) and 4 (EAGLE), with EAGLE achieving 1.99× throughput. The paper frames this as "EAGLE achieves a 2x increase in throughput" (Section 4.4).

**Tree attention tradeoff at batch size > 1.** The paper notes that "at bs=7, the computational resources are less abundant, making the non-use of tree attention more advantageous" (Section 4.4). This is a practical engineering consideration: when GPU compute is saturated, the additional token processing from tree attention can increase latency more than it increases acceptance length, potentially reducing net speedup. The paper does not report separate speedup numbers with and without tree attention at different batch sizes, but the observation suggests that the optimal tree structure (or the decision to use tree attention at all) may depend on batch size and available compute headroom.

#### Integration with gpt-fast: EAGLE + Quantization + Compilation

Table 4 (Section 4.2) demonstrates composability with orthogonal acceleration methods. Combining EAGLE with gpt-fast (which uses quantization and torch.compile for kernel fusion) on LLaMA2-Chat 7B:

- Vanilla HuggingFace: 24.5 tokens/s (FP16)
- gpt-fast alone: 55.1 tokens/s (FP16), 106.9 tokens/s (int4)
- EAGLE + gpt-fast: **100.2 tokens/s** (FP16), **160.4 tokens/s** (int4)

The combined system (EAGLE + gpt-fast at int4) achieves 160.4 tokens/s, which is 6.55× faster than vanilla HuggingFace (24.5 tokens/s) and 1.50× faster than gpt-fast alone at int4 (106.9 tokens/s). The speedup ratio of EAGLE on top of gpt-fast (100.2 / 55.1 = 1.82× at FP16, 160.4 / 106.9 = 1.50× at int4) is lower than EAGLE's standalone speedup (2.78× on LLaMA2-Chat 7B from Table 1), because gpt-fast already reduces the per-token latency, leaving less room for speculative sampling to amortize. Nevertheless, the composability demonstrates that EAGLE's benefits multiply with—rather than being subsumed by—lower-level optimizations.

The paper frames this as evidence that "EAGLE operates in parallel with other acceleration or throughput-improving methods, such as quantization, compilation, etc. Combining EAGLE with these techniques could further reduce the operational costs of LLM systems" (Section 4.1).

---

### Ablation Studies and Robustness Checks

**Tree attention vs. chain-structured drafts (Section 4.3.1, Table 5, Figure 7):** Using tree-structured drafts (with the specific branching structure in Figure 9) increases average acceptance length τ by 0.62–0.75 across Vicuna and LLaMA2-Chat models (e.g., Vicuna 7B: 3.20 → 3.94, Vicuna 33B: 2.97 → 3.68). The speedup ratio increases by approximately 0.3×–0.5× (Figure 7), with the improvement being smaller than the τ increase "because tree draft and verification do not increase the number of forward passes in the model... but they do increase the number of tokens processed per forward pass," adding computational cost that partially offsets the acceptance-length gain. Notably, even without tree attention, EAGLE achieves "a significant acceleration effect, approximately in the range of 2.3x-2.7x" (Section 4.3.1), demonstrating that the core feature+shifted-token mechanism provides most of the benefit independent of the tree structure.

**Input representations: feature&shifted-token vs. alternatives (Section 4.3.2, Figure 8):** This is the paper's central ablation, isolating the contributions of feature-level prediction and shifted-token conditioning. On Vicuna 7B at temperature=0 on MT-bench:
- **token-only input:** ~1.5× speedup (baseline: standard token-level autoregression)
- **feature-only input:** ~1.9× speedup (feature-level autoregression without uncertainty resolution)
- **feature&unshifted-token input:** ~2.1× speedup (features plus aligned tokens, providing semantic context but not sampling outcomes)
- **feature&shifted-token input (EAGLE):** ~2.8× speedup (full EAGLE with uncertainty resolution)

Three conclusions are drawn from the accompanying acceptance-rate plots (Figure 8, bottom rows): (1) "When the number of parameters of the draft model is limited, utilizing features yields slightly better results than tokens" (feature-only > token-only). (2) "Merging features and tokens modestly boosts performance, mainly as discrete, error-free tokens mitigate feature error accumulation, evident from the similar 0-α of feature&unshifted-token and feature-only draft models, with a significantly improved 1-α" (feature&unshifted-token > feature-only, with the gain concentrated in error-accumulation scenarios). (3) "Addressing the randomness inherent in the sampling process results in the most significant improvement. The feature&shifted-token scheme, compared to feature&unshifted-token, adds no complexity yet markedly enhances the draft model's capability by simply advancing the token by one time step" (feature&shifted-token > feature&unshifted-token, the largest single jump). The same patterns are replicated at temperature=1 (Figure 8, middle row), confirming the finding is not specific to greedy decoding.

**Training data source: fixed dataset vs. target-LLM-generated data (Section 4.3.3, Table 6):** Training EAGLE on data generated by LLaMA2-Chat 7B autoregressively (same ShareGPT questions, target-LLM-generated answers) yields a speedup ratio of 2.88× vs. 2.78× using the fixed ShareGPT dataset, with τ = 3.75 vs. 3.62. The small difference (0.10× speedup, 0.13 in τ) indicates that "EAGLE exhibits low sensitivity to training data" and justifies the fixed-dataset approach for cost reduction. This is a practically significant negative result: the expensive step of generating on-policy training data from the target LLM is not necessary.

**Batch size scaling (Section 4.4, Table 7):** As discussed under Main Results, the speedup ratio decreases from 2.90× (bs=1) to 2.40× (bs=4) for LLaMA2-Chat 70B. This is an expected consequence of LLM inference being memory-bound: larger batch sizes more fully utilize the GPU, reducing the spare computational capacity that speculative sampling exploits. The paper does not report speedups for batch sizes beyond 4.

---

### Critical Assessment

The experimental section is thorough in breadth—testing across six model sizes, four tasks, two temperatures, and multiple batch sizes—but has specific limitations that affect the strength of the paper's claims.

**Claim: EAGLE achieves 2.7×–3.5× speedup on LLaMA2-Chat 70B.** The experiments clearly demonstrate speedup ratios in this range for the specific configurations tested (MT-bench at batch size 1, temperature=0). Figure 1 shows 3.01× for LLaMA2-Chat 70B; Table 1 shows 3.52× on HumanEval and 3.03× on GSM8K for the same model. However, these numbers are reported without confidence intervals, without multiple runs, and without variance estimates. For temperature=1, the speedup ratios are lower (2.67× on MT-bench, 2.92× on HumanEval). The 2.7×–3.5× range therefore holds for greedy decoding on relatively structured tasks (code, math, multi-turn dialogue), but the paper should be more precise about which settings achieve the upper vs. lower end of the range. The highest reported number (3.76× on LLaMA2-Chat 13B for HumanEval) and the lowest (2.13× on Vicuna 7B at temperature=1, Figure 2) define the full span of results—the 2.7×–3.5× claim is a fair characterization of the typical greedy-setting results but not a universal guarantee.

**Claim: EAGLE achieves 1.7×–2.1× speedup over Lookahead and 1.5×–1.6× over Medusa.** These comparisons rely on speedup numbers copied from the Medusa and Lookahead technical reports. The paper does not re-evaluate these methods on the same hardware, with the same software stack, or under identical conditions. Hardware differences (GPU type, CUDA version, PyTorch version, inference framework) can substantially affect speedup ratios, which are measured as walltime reductions. For instance, if Medusa's numbers were measured on a different GPU generation or with different attention kernel implementations, the comparison may not be apples-to-apples. The paper does not discuss hardware parity or control for it. A stronger evaluation would have reproduced Medusa and Lookahead on the same hardware as EAGLE, or at minimum documented the hardware differences and their potential impact. The comparison to standard speculative sampling and DistillSpec is more reliable because these were evaluated by the paper itself (Figures 1 and 2) rather than taken from external reports.

**Claim: Distribution preservation is "theoretically guaranteed."** The paper does not empirically verify that EAGLE's output distribution matches vanilla decoding. While the theoretical guarantee from Leviathan et al. (2023) applies to any properly-implemented speculative sampling procedure, verifying that EAGLE's implementation is correct would require statistical distributional tests (e.g., comparing token frequency distributions, measuring KL divergence between EAGLE-generated and vanilla-generated text on a large sample). The paper states that "evaluating the quality of EAGLE's generated results is both unnecessary and meaningless" because of the theoretical guarantee (Section 4), but this conflates "unnecessary in theory" with "unnecessary in practice." Implementation bugs—incorrect acceptance/rejection logic, off-by-one errors in the multi-round tree sampling, numerical precision issues—could violate the guarantee in practice without being obviously visible in speedup measurements. No quality evaluation of any kind is reported, which is a missing validation step given the complexity of the tree-structured verification algorithm (Algorithm 1).

**Missing ablation: Draft model depth and width.** The paper uses a single transformer decoder layer as the Autoregression Head but never evaluates how performance scales with draft model capacity. Would two decoder layers improve speedup enough to justify the additional overhead? Would a smaller hidden dimension or fewer attention heads reduce memory footprint without significant accuracy loss? The choice of exactly one decoder layer is not ablated, making it unclear whether this is near-optimal or simply a conservative default.

**Missing ablation: Tree structure optimization.** The tree structure (Figure 9) is described as "not rigorously optimized but rather based on intuition" (Appendix A.1). How sensitive is the speedup to the choice of branching factors and depth? The paper acknowledges that "the optimal tree structure is likely context-dependent" and that "tuning the draft structure could potentially lead to improved performance," but does not explore this dimension experimentally. For a method where tree attention provides 0.3×–0.5× additional speedup (Section 4.3.1), understanding how much further gain might come from optimized tree shapes is relevant to assessing the method's ceiling.

**Training data: 68K dialogues from ShareGPT only.** While the paper shows low sensitivity to data source (fixed vs. target-LLM-generated, Table 6), it does not explore sensitivity to data quantity or domain. Would training on 10K dialogues achieve similar performance? Would 500K dialogues improve speedup meaningfully? Is ShareGPT—a specific dataset of ChatGPT conversations with a particular style—representative enough that EAGLE trained on it transfers to GSM8K math reasoning or HumanEval code generation? The paper notes that all experiments "employ the same weights, trained exclusively on the ShareGPT dataset, without any additional training on the evaluation datasets" (Section 1), which demonstrates cross-domain transfer, but the paper does not ablate whether domain-matched training data would improve performance on specific tasks (e.g., training on code for HumanEval, on math for GSM8K).

**Single hardware configuration for most experiments.** The paper's primary results are on unspecified GPU hardware (likely A100, since training uses A100 40G). The batch-size experiments use an RTX 3090. Different GPUs have different memory bandwidth, compute capacity, and FLOPs-to-bandwidth ratios—all of which affect the speedup achievable through memory-bound speculative sampling. A GPU with higher memory bandwidth relative to compute (e.g., H100 vs. A100) might show different speedup ratios because the spare computational capacity that EAGLE exploits differs. The paper does not report sensitivity to hardware.

**The 2× throughput claim is based on reduced maximum batch size.** EAGLE's throughput improvement (Table 7) is computed at the maximum batch size possible under a memory constraint, which is lower for EAGLE (bs=7) than for vanilla decoding (bs=8) on the RTX 3090. The 1.97× throughput gain therefore comes from measuring EAGLE at bs=7 vs. vanilla at bs=8—a comparison that gives EAGLE a slight disadvantage in batch size but still shows near-2× improvement. For LLaMA2-Chat 70B, the comparison is EAGLE at bs=4 vs. vanilla at bs=5, yielding 1.99× throughput. This is a fair comparison under realistic memory constraints, but the reader should understand that at identical batch sizes, the throughput gain might differ. The paper does not report throughput at fixed batch sizes (e.g., both methods at bs=4), which would provide a cleaner comparison.

**No latency breakdown.** The paper reports end-to-end walltime speedup but does not break this down into draft-model overhead, verification overhead, and acceptance computation. Understanding where the time is spent would be diagnostic for identifying bottlenecks and guiding future improvements. For instance, if the draft model's forward passes consume 40% of the total inference time, then further draft-model optimization would be a high priority; if verification accounts for most of the cost, then reducing tree size or optimizing tree attention would be more impactful. The lack of a latency breakdown is a missed opportunity for deeper analysis.

**Theoretical guarantee not empirically validated on non-greedy.** For temperature=1, the paper reports speedup ratios but does not verify that the output distribution is preserved. While the theoretical guarantee applies, the tree-structured multi-round sampling algorithm (Algorithm 1) is non-trivial to implement correctly, and subtle bugs in the recursive acceptance/rejection logic could produce tokens from a slightly different distribution. A simple validation—generating a large corpus with EAGLE and with vanilla decoding, and comparing token frequencies or n-gram distributions—would have strengthened confidence in the implementation. The absence of such validation, combined with the paper's explicit claim that Medusa does not guarantee lossless non-greedy performance, makes the distribution-preservation claim rest entirely on the theoretical proof rather than empirical demonstration.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Overhead Is Not Accounted for in the Reported Speedup

**The assumption or constraint.** EAGLE's drafting process requires, at each step, both the token sequence $T_{2:i+1}$ (the tokens advanced by one time step) and the feature sequence $F_{1:i}$ (the target LLM's second-to-top-layer hidden states). The features for all context tokens must be computed by the target LLM before drafting can begin—most critically, the feature $f_i$ for the most recent token and the feature sequence for the entire prompt prefix. The paper assumes these features are available at negligible cost because they are produced during the verification phase of the previous drafting round (Section 3.3: "we document accepted tokens and their features for use in the next drafting phase"). However, this means EAGLE's per-round inference latency must include the verification forward pass that produces these features, which is already accounted for. The unaccounted cost is the **feature extraction for the prompt itself**—before the first drafting round, the target LLM must process the entire prompt in a standard forward pass to produce $F_{1:N}$ for all $N$ prompt tokens. The paper does not discuss whether this initial prompt-processing cost is amortized over subsequent generation or whether it imposes a fixed overhead that reduces speedup for short responses.

**The consequence.** For applications with long prompts and short responses (e.g., retrieval-augmented generation where a large document is prepended, or classification tasks where the output is a single token), the initial feature extraction pass over the full prompt could dominate total inference time, making EAGLE's per-token generation speedup largely irrelevant to end-to-end latency. The paper's speedup ratios are measured on MT-bench, HumanEval, GSM8K, and Alpaca, where typical prompt lengths are moderate (a few hundred tokens for MT-bench multi-turn dialogues, shorter for HumanEval and GSM8K). In settings where the prompt-to-response length ratio is much higher, the headline speedup numbers would not reflect actual end-to-end experience. More subtly, EAGLE **adds a memory cost for storing features**—the feature vectors for all prompt and generated tokens must be retained in GPU memory (each of dimension `hidden_dim` in FP16, so 8 KB per token for a 4096-dim model), which for very long contexts (e.g., 100K+ tokens) could become a significant memory burden not present in vanilla decoding.

**What evidence exists in the paper.** The paper does not measure or discuss prompt-processing overhead, feature storage memory cost, or speedup as a function of prompt length. All speedup experiments use the standard benchmarks' prompt lengths without ablation. The paper acknowledges the memory overhead of the draft model itself (Section 4.4: maximum batch size decreases from 8 to 7 for Vicuna 7B on a 24 GB GPU), but does not extend this analysis to feature storage overhead for long contexts.

**Mitigation status.** Not addressed. The paper does not propose any mechanism for reducing prompt-processing cost (e.g., reusing features from the prompt processing as draft-model inputs for the first drafting round, or compressing feature sequences for long contexts). The feature extraction pass is inherent to EAGLE's design—the draft model needs features to autoregress—but the paper treats it as an unremarked prerequisite rather than a cost to be analyzed or optimized.

---

### The Speedup Ratio Is Highly Hardware-Dependent and Not Guaranteed to Transfer

**The assumption or constraint.** EAGLE's speedup is fundamentally based on exploiting **spare GPU computational capacity** that is unused during vanilla autoregressive decoding due to the memory-bound nature of LLM inference (Section 4.4: "Inference in LLMs is memory-bound, leaving GPU computational resources underutilized. The principle behind the speculative sampling-based approach in enhancing generation speed lies in more effectively utilizing GPU computational resources."). This means the achievable speedup depends critically on the **ratio of compute to memory bandwidth** on the specific GPU hardware. A GPU with abundant compute relative to memory bandwidth (e.g., H100 with HBM3) will show larger speedups because there is more spare compute for the draft model to utilize. A GPU with tighter compute-to-bandwidth ratios (e.g., older architectures, edge devices, or CPUs) will show smaller speedups—potentially no speedup at all if the draft model's additional compute requirements stall the memory subsystem further. The paper reports all main results on unspecified hardware (likely A100 GPUs, given training used "4x A100 (40G)"), with the batch-size experiments on an RTX 3090. There is no evaluation across GPU generations, architectures, or compute tiers.

**The consequence.** A practitioner deploying EAGLE cannot assume they will achieve the reported 2.7×–3.5× speedup on their hardware. On an H100, the speedup might be higher; on an A10 or T4 (common in cloud inference), it could be substantially lower. On CPU inference, where compute is the bottleneck rather than memory bandwidth, EAGLE might achieve zero or negative speedup because the draft model adds computation without exploiting spare capacity. The paper acknowledges this principle explicitly for batch size scaling (Section 4.4: as batch size increases, GPU compute utilization rises, reducing speedup), but does not extend the analysis to hardware variation. The throughput results under memory constraints (Table 7) provide a partial view of one dimension (memory capacity), but the compute-bandwidth ratio—which determines the very existence of spare capacity—is not analyzed.

**What evidence exists in the paper.** Table 7 and Section 4.4 provide the only hardware-related evidence: speedup decreases from 3.01× (bs=1) to 2.40× (bs=4) on LLaMA2-Chat 70B, confirming sensitivity to GPU utilization. The gpt-fast integration (Table 4) provides indirect evidence: on an RTX 3090, EAGLE + gpt-fast achieves 1.50×–1.82× speedup over gpt-fast alone, compared to 2.78× standalone EAGLE speedup on LLaMA2-Chat 7B—demonstrating that when the baseline inference is already optimized (reducing spare capacity), EAGLE's relative benefit shrinks. The Mixtral 8x7B result (1.50×, Table 3) provides architectural sensitivity evidence: the MoE's different compute-to-parameter-access pattern reduces speedup dramatically. But there is no direct comparison of the same model across different GPU types.

**Mitigation status.** Not addressed. The paper does not report hardware specifications for its main experiments, does not test on multiple GPU architectures, and does not provide guidance for practitioners on expected speedup as a function of hardware characteristics. The authors frame EAGLE as operating "in parallel with other acceleration or throughput-improving methods, such as quantization, compilation, etc." (Section 4.1), but do not discuss how the composition of methods affects the total available spare compute that EAGLE depends on. A practitioner combining EAGLE with FlashAttention, quantization, and compilation might find that the cumulative optimizations saturate GPU compute, leaving minimal headroom for EAGLE—the gpt-fast results (Table 4) already show the speedup ratio shrinking from 2.78× to 1.50× when combined with aggressive lower-level optimizations.

---

### The Tree Structure Is Not Optimized and May Be Suboptimal for Many Settings

**The assumption or constraint.** EAGLE uses a fixed tree structure (Figure 9, Appendix A.1) for all models, all tasks, all temperatures, and all batch sizes. The structure—branching factors of 4, 3, 2, 1 across five levels—was chosen "not rigorously optimized but rather based on intuition: branches of higher-probability tokens should be deeper and wider" (Appendix A.1). The paper explicitly acknowledges that "the optimal tree structure is likely context-dependent. For instance, as batch size increases and redundant computational resources decrease, a smaller tree might be preferable. Tuning the draft structure could potentially lead to improved performance" (Appendix A.1). This means the reported speedup ratios are conditioned on a specific tree topology that was not systematically searched, and the method's true ceiling—with an optimized tree—is unknown.

**The consequence.** At batch size > 1, the paper notes that "at bs=7, the computational resources are less abundant, making the non-use of tree attention more advantageous" (Section 4.4). This suggests that the fixed tree structure is already suboptimal for larger batch sizes—the additional tokens processed per verification pass cost more in latency than they gain in acceptance length. Similarly, for tasks where the draft model is highly accurate (code generation, with τ = 4.29–4.52), a deeper tree with fewer branches might produce longer accepted sequences at lower verification cost; for tasks with lower accuracy (Alpaca, τ = 3.61–3.86), a shallower, wider tree might be better. The fixed tree also ignores per-prompt difficulty: for prompts where the model is confident (high draft accuracy), a more aggressive tree could capture more tokens; for uncertain prompts, a conservative tree avoids wasted computation. The paper's results therefore represent a **lower bound** on achievable speedup for any given configuration, but the gap between the fixed-tree speedup and the optimized-tree speedup is unknown—it could be negligible or substantial.

**What evidence exists in the paper.** The tree vs. chain ablation (Section 4.3.1, Table 5, Figure 7) demonstrates that adding tree attention increases speedup by approximately 0.3×–0.5× over chain-structured drafts, establishing that the tree structure matters. The observation about batch size 7 (Section 4.4) provides direct evidence that the fixed tree is suboptimal in at least one regime. The n-α metrics (Table 2) provide indirect evidence of structure sensitivity: the decrease from 1-α to 4-α (e.g., 0.74 → 0.67 for Vicuna 7B) indicates that deeper branches in the tree (which accumulate more feature prediction errors) have progressively lower acceptance rates, suggesting that the optimal tree depth depends on the draft model's error propagation characteristics. But no ablation varies the tree topology—not depth, not branching factors, not the shape of the fan-out.

**Mitigation status.** The paper acknowledges the limitation explicitly (Appendix A.1) and suggests tuning as future work, but provides no guidance on how to tune (search procedure, objectives, constraints) or estimates of the potential gain. The lack of even a coarse sweep over tree structures (e.g., {2, 3, 4, 5} depth, {2, 3, 4} branching factors) means the reader cannot assess whether the fixed structure is near-optimal or heavily suboptimal. For a method where the tree structure is described as "a key component" (tree attention increases τ by 0.6–0.8, Table 5), the absence of structure optimization is a notable gap in the experimental analysis.

---

### Distribution Preservation Is Claimed Theoretically but Never Validated Empirically

**The assumption or constraint.** The paper's central quality guarantee is that EAGLE preserves the target LLM's output distribution exactly: "EAGLE does not involve any fine-tuning of the original LLM, and the preservation of the output distribution by EAGLE is theoretically guaranteed for both the greedy and non-greedy settings" (Section 1). This guarantee relies on the speculative sampling proof from Leviathan et al. (2023), which applies to EAGLE's verification procedure (Algorithm 1) in principle. However, the proof's validity depends on a correct implementation of the multi-round tree-structured acceptance/rejection algorithm—which is substantially more complex than standard chain-structured speculative sampling, involving recursive adjusted distributions, tree attention masking, and proper handling of all sibling branches at each depth. The paper provides no empirical validation that the implementation is correct, relying entirely on the theoretical proof.

**The consequence.** Any implementation bug in Algorithm 1—incorrect adjusted distribution computation, off-by-one errors in the recursive calls, numerical precision issues in the acceptance probability calculation ($\min(1, p/\hat{p})$), incorrect handling of the tree attention mask during verification—could cause EAGLE to produce tokens from a distribution that differs from vanilla decoding. Such bugs would not be visible in speedup measurements or acceptance rate metrics (since both are computed against whatever tokens the buggy algorithm produces). The paper explicitly declines to evaluate output quality: "evaluating the quality of EAGLE's generated results is both unnecessary and meaningless" (Section 4). While this position is defensible in theory (a correct implementation of a proven algorithm needs no validation), it conflates the algorithm's theoretical properties with the implementation's factual correctness. The recursive adjusted-distribution logic in Algorithm 1—where after each rejection, the target distribution is updated as $p \leftarrow \text{norm}(\max(0, p - \hat{p}))$ and the next sibling is tested against this modified distribution—is particularly delicate. If $\max(0, p - \hat{p})$ is not correctly normalized, or if the subtraction introduces floating-point errors that accumulate across multiple rejections at the same position, the resampled token's distribution may deviate from the target.

**What evidence exists in the paper.** None. The paper includes no text quality evaluation, no distributional comparison, no KL divergence measurement, and no statistical test comparing EAGLE-generated text to vanilla-generated text. The acceptance rate metrics (Table 2, Table 8) measure draft quality, not output quality—they count how often draft tokens are accepted, not whether the accepted tokens match what vanilla decoding would have produced. The speedup ratios measure latency, not fidelity. The paper provides the pseudocode for Algorithm 1 (Appendix A.2) and invokes the Leviathan et al. (2023) proof, but does not empirically demonstrate that the implemented algorithm matches the proven one.

**Mitigation status.** Not addressed. The authors could have included a simple validation: generate a large corpus (e.g., 10,000 completions of MT-bench prompts) with both EAGLE and vanilla decoding, and compare token frequency distributions, n-gram distributions, or perform a two-sample statistical test for distributional equality. This would have required minimal additional computation (since MT-bench experiments were already running) and would have provided empirical confidence in the implementation. The absence of such validation is particularly notable given that the paper explicitly criticizes Medusa for not guaranteeing lossless non-greedy performance (Figure 2 caption)—a criticism that implicitly promises EAGLE's guarantee is real, not just theoretical. For a method whose primary value proposition over Medusa and Lookahead is the lossless distribution guarantee, failing to validate that the guarantee holds in practice is a significant omission.

---

### Performance on Hard or Unpredictable Text Is Not Characterized

**The assumption or constraint.** EAGLE's speedup depends on the draft model's ability to predict the target LLM's next features accurately. This ability varies with the **predictability of the generated text**: code with fixed templates is highly predictable (HumanEval: τ = 4.42–4.52), while open-ended dialogue and instruction-following are less predictable (MT-bench: τ = 3.62–3.98, Alpaca: τ = 3.61–3.86). The paper characterizes performance on four specific task types, but does not explore the full spectrum of text predictability that a deployed LLM might encounter—from highly structured (JSON output, SQL queries, translation) to highly unpredictable (creative writing, poetry, humor, brainstorming). The paper's conclusion that the best performance occurs on code "due to the prevalence of fixed templates" (Section 4.1) implies that the worst performance would occur on tasks with minimal template structure, but no such tasks are evaluated.

**The consequence.** A practitioner deploying EAGLE for a general-purpose chatbot—where user queries span the full range from structured fact-retrieval to open-ended creative requests—cannot assume uniform speedup. Some queries will benefit from 3×+ speedup (those that elicit predictable, template-like responses); others may benefit very little (those that elicit unpredictable, high-entropy responses where the draft model struggles to anticipate the target LLM's features). The paper does not provide per-query speedup distributions, minima, or variance estimates—only average speedup across the benchmark. If the distribution is heavy-tailed (many queries near the average, but a non-trivial fraction far below it), the user experience would be inconsistent, with some responses arriving quickly and others at near-vanilla latency. This matters for latency-SLA-bound applications where worst-case, not average-case, latency determines system design.

More fundamentally, the paper does not characterize what properties of a prompt or response make it predictable versus unpredictable for EAGLE. Is it lexical repetition? Syntactic regularity? Low perplexity under the draft model? Without this analysis, a practitioner cannot predict whether their specific use case will benefit from EAGLE without running their own benchmarks.

**What evidence exists in the paper.** The task-level breakdown (Table 1) provides a coarse view: HumanEval (code) > GSM8K (math) ≈ MT-bench (dialogue) > Alpaca (instruction-following) in terms of τ and speedup. But within each task, no per-example analysis is provided. The n-α metrics (Table 2) show that the draft model's performance degrades gradually with error accumulation (0-α through 4-α decline only modestly), but this measures robustness to the draft model's own errors, not sensitivity to input predictability. The paper does not report the correlation between text perplexity (under the target LLM) and EAGLE's speedup, which would be a natural diagnostic.

**Mitigation status.** Not addressed. The paper does not discuss worst-case latency, per-query variance, or the relationship between text predictability and speedup. The fixed tree structure (Appendix A.1) is used uniformly across all queries, making no adaptation to per-query predictability—a query that is clearly predictable (the draft model's top-1 token has probability 0.99) gets the same tree as one that is highly uncertain (the distribution is flat across many tokens), even though the optimal draft strategy likely differs. This is a missed opportunity for dynamic adaptation that could both improve average speedup and reduce variance.

---

### The Method Adds Non-Trivial GPU Memory Overhead, Limiting Maximum Batch Size and Throughput Scaling

**The assumption or constraint.** EAGLE requires storing the draft model's parameters (0.24B–0.99B, depending on target model size) in GPU memory alongside the target LLM. This additional memory consumption reduces the maximum batch size possible under a fixed memory budget. The paper reports (Section 4.4) that on a 24 GB RTX 3090, the maximum batch size for Vicuna 7B decreases from 8 (vanilla) to 7 (EAGLE); on 4× A100 40G (160 GB total), the maximum batch size for LLaMA2-Chat 70B decreases from 5 to 4. Under these memory-constrained maximum batch sizes, EAGLE achieves approximately 2× throughput improvement (1.97× for Vicuna 7B, 1.99× for LLaMA2-Chat 70B, Table 7). However, the throughput comparison is made at **different batch sizes** (vanilla at its maximum, EAGLE at its maximum), not at a fixed batch size. At a fixed batch size of, say, 4 on LLaMA2-Chat 70B (which both methods can support), the throughput improvement might be different—potentially higher (if EAGLE's latency advantage is fully expressed) or lower (if the memory overhead of EAGLE's draft model displaces other optimizations).

**The consequence.** In throughput-constrained serving scenarios—where the goal is to maximize tokens per second per GPU rather than minimize per-query latency—EAGLE's benefit is substantially smaller than the headline latency speedup suggests. A 3× latency speedup (3.01× on LLaMA2-Chat 70B at bs=1, Figure 1) translates to only 2× throughput improvement at maximum batch size (1.99×, Table 7). This gap arises from two factors: (1) the draft model's memory consumption forces a reduction in batch size, partially offsetting the per-query speedup, and (2) at larger batch sizes, GPU compute utilization is higher, reducing the spare capacity that EAGLE exploits. For practitioners running high-throughput batch inference (e.g., generating embeddings, scoring large document collections, or serving many concurrent users), the 2× throughput figure—not the 3× latency figure—is the relevant metric, and it represents a more modest practical gain.

Moreover, the memory overhead scales with the target model's hidden dimension. For LLaMA2-Chat 70B (hidden_dim = 8192), EAGLE adds 0.99B parameters—nearly 2 GB in FP16. For even larger models (e.g., LLaMA 405B or future models), the draft model's parameter count would grow proportionally to hidden_dim², potentially becoming a significant fraction of the target model's own memory footprint. The paper does not discuss how EAGLE's memory overhead scales with model size or whether the draft model's capacity can be reduced for larger targets without sacrificing accuracy.

**What evidence exists in the paper.** Table 7 and Section 4.4 provide the batch-size and throughput measurements. The maximum batch size reduction (8→7 for Vicuna 7B, 5→4 for LLaMA2-Chat 70B) is reported transparently. The 2× throughput figure is computed fairly as the ratio of tokens-per-second at each method's respective maximum batch size. However, the paper does not report a throughput vs. batch-size curve (showing throughput for both methods at each feasible batch size) or analyze the scaling behavior of draft-model memory with target model size.

**Mitigation status.** Partially addressed through transparency—the paper reports the batch-size reduction and does not hide it. But the paper does not explore mitigation strategies: quantization of the draft model (whose 0.24B–0.99B parameters could potentially be stored in int8 or int4 without significant accuracy loss, since the draft model is only one decoder layer), sharing of attention KV-caches between the target LLM and draft model, or offloading the draft model to CPU when not in use. The composability with gpt-fast (Table 4) demonstrates that EAGLE works alongside quantization, but this applies to the target LLM, not the draft model specifically. The paper frames the throughput result positively ("EAGLE achieves a 2x increase in throughput") without emphasizing the gap between this figure and the headline 3× latency speedup, which could mislead practitioners who primarily care about throughput.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that “lossless” acceleration can be pushed further by moving drafting to the feature space and explicitly resolving sampling uncertainty with shifted tokens. This reframes speculative decoding design space away from token prediction toward internal state prediction (Sections 1, 3; Figures 3–4).
- Practical applications:
  - Low-latency chat systems, code assistants, math solvers, and instruction followers that require the exact same output distribution as the baseline model (Abstract; Tables 1–2). 
  - Production deployments can combine EAGLE with quantization/compilation (e.g., gpt-fast) for additive gains—up to 160.4 tokens/s on LLaMA2-Chat 7B int4 on a single RTX 3090 (Table 4).
- Follow-up research:
  - Auto-tuning tree structures and branching policies for different workloads and hardware (Appendix A.1 notes this as future optimization).
  - Extending the “feature + shifted-token” idea to multi-modal, retrieval-augmented, or structured decoding settings.
  - Investigating better robustness to feature prediction errors (e.g., multi-step consistency losses, teacher-forced vs scheduled sampling in feature space).
  - Specialized designs for MoE verification to avoid reading many experts per pass, improving MoE speedups (Table 3 discussion).
  - Exploring partial fine-tuning of early layers to produce more predictable second-to-top features without changing the LM head distribution.

> Core takeaway: By predicting the next internal feature conditioned on the realized next token and verifying a token tree with a lossless speculative sampler, EAGLE turns each target-LM pass into ≈3–4 accepted tokens on average (Tables 1–2), achieving consistent 2–4x latency gains while keeping the output distribution identical to vanilla decoding (Sections 2–3; Figures 1–2).
