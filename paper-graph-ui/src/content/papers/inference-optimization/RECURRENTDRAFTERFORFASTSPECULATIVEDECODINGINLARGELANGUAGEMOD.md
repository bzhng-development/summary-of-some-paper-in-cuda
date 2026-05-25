# RECURRENT DRAFTER FOR FAST SPECULATIVE DECODING IN LARGE LANGUAGE MODELS

**ArXiv:** [2403.09919](https://arxiv.org/abs/2403.09919)

## 🎯 Pitch

Recurrent Drafter (ReDrafter) introduces a novel speculative decoding technique that pairs large language models with an efficient RNN-based drafter, enabling the generation and verification of multiple tokens per LLM forward pass—dramatically speeding up inference without sacrificing exactness of outputs. By leveraging a GPU-friendly dynamic tree attention to eliminate redundant computation and training the drafter through knowledge distillation, ReDrafter achieves state-of-the-art acceleration (up to 2.8× on GPUs, 2.3× on Apple Silicon) while preserving model fidelity. This innovation is key for real-time and resource-constrained applications, slashing latency and infrastructure costs while keeping outputs identical to the original LLM.

---

## 1. Executive Summary

This paper introduces **Recurrent Drafter (ReDrafter)**, a speculative decoding approach that accelerates LLM inference by using a lightweight recurrent neural network draft model conditioned on the LLM's hidden states, applying beam search with a dynamic tree attention algorithm to eliminate duplicated prefixes among candidate sequences, and training the draft model via knowledge distillation from the LLM. ReDrafter achieves up to 2.8× speedup for Vicuna models on MT-Bench with a PyTorch implementation on Nvidia H100 GPUs and up to 2.3× speedup on-device with an MLX implementation on Apple Silicon Metal GPUs, establishing that an RNN-based drafter with beam search can effectively convert computational resources into speedups across both server-grade and resource-constrained hardware, with optimal beam width depending critically on the available GPU compute capacity.

## 2. Context and Motivation

### The Core Problem: LLMs Are Memory-Bandwidth Bound at Inference Time

To understand why ReDrafter matters, you need to understand the fundamental bottleneck in serving large language models. When an LLM generates text — one token at a time, autoregressively — each forward pass through the model reads every single parameter from GPU memory into the compute units, performs the matrix multiplications and attention operations required to predict the next token, and writes results back. The computational intensity (the ratio of FLOPs performed to bytes of data moved) is surprisingly low. This means the GPU's arithmetic units spend most of their time waiting for data to arrive from memory, rather than actually computing.

This is the **memory-bandwidth bottleneck** that the paper identifies in Section 2:

> "It is widely recognized that LLM generation is constrained by the memory bottleneck."

Since model weights must be read from memory for every token generated — and those weights can be tens of billions of parameters (hundreds of gigabytes) — the throughput of inference is fundamentally limited by how fast memory can feed the compute units, not by how fast those units can multiply numbers. Increasing batch size helps because reading weights once amortizes across multiple sequences in a batch, but many interactive applications (chatbots, code assistants, on-device assistants) operate at batch size 1, where this bottleneck is most severe.

### The Gap: Existing Speculative Decoding Designs Face a Compute-Efficiency Tradeoff

The core idea of speculative decoding — introduced by Leviathan et al. (2023) and Spector & Re (2023) — addresses this bottleneck by shifting the workload from memory-bound to compute-bound. Instead of having the LLM generate every token sequentially, a smaller, faster **draft model** predicts several candidate tokens at once. The LLM then verifies the entire candidate sequence in a single forward pass, accepting the longest prefix that matches its own prediction. If the draft model is accurate and the verification step accepts multiple tokens per LLM call, you reduce the total number of LLM forward passes — and thus reduce the total memory access — producing a net speedup even though you're doing *more* total computation (the draft model's work plus verification).

However, the design of the draft model is where the field had an unresolved tension, which ReDrafter directly addresses. Prior approaches fell into two camps, each with significant limitations:

**Camp 1: Separate draft models detached from the LLM.** This is the approach of Leviathan et al. (2023), Chen et al. (2023), and Sun et al. (2024). You train a small, independent language model to approximate the LLM's outputs, and at inference time you run both models. The advantage is simplicity and modularity — especially attractive when an off-the-shelf smaller model from the same family (e.g., LLaMA 7B drafting for LLaMA 70B) already approximates the target. The disadvantage, as the paper notes in Section 2, is twofold:

> "If no off-the-shelf candidate is available, the draft model must be trained separately from the LLM, with efforts focused on aligning it as closely as possible to the LLM. Additionally, deploying two separate models adds complexity to their integration within a unified serving system."

The alignment problem is particularly thorny: a separately trained draft model doesn't have access to the LLM's internal representations, so it must independently model the probability distribution of next tokens from scratch. This is inherently less sample-efficient than a model that can directly condition on the LLM's hidden states.

**Camp 2: Draft heads attached to the LLM's hidden states.** Medusa (Cai et al., 2024) pioneered an elegant alternative: instead of a separate model, attach multiple lightweight prediction heads directly to the LLM's intermediate hidden states. Each head is responsible for predicting a token at a specific future position — head 1 predicts token $t+1$, head 2 predicts token $t+2$, and so on. Because these heads condition on the LLM's rich internal representations, they can be much smaller than a standalone draft model and achieve reasonable accuracy.

But this approach has two problems that ReDrafter identifies and addresses:

**Problem 1: Independent predictions ignore sequential structure.** Medusa's heads predict each future token independently — they don't know what the adjacent heads predicted. The paper explains:

> "Its independent prediction mechanism does not leverage the sequential structure, resulting in limited predictive accuracy and an exponentially large set of feasible candidate token sequences."

Concretely, if head 1 predicts "morning" and head 2 predicts "sipping," there's no mechanism ensuring these form a coherent continuation. Medusa compensates by considering all $|V|^T$ possible combinations across its $T$ heads (where $|V|$ is vocabulary size), pruning this exponentially large space via tree attention. But the underlying accuracy limitations remain: without sequential conditioning, the predictions at later positions degrade rapidly.

**Problem 2: Computational resources are not efficiently converted to speedups.** The paper makes a pointed observation about Medusa's approach:

> "This excessive computational effort may not yield proportional speedups, as independent predictions become less accurate as $T$ increases, resulting in suboptimal predictive accuracy and a lower acceptance rate from the LLM."

In other words, throwing more parallel compute at the problem (more heads, more predictions) doesn't necessarily help if those predictions are inaccurate, because the LLM verification step will reject them, wasting the draft computation.

**Camp 2.5: Recurrent draft models with limited parallelism.** EAGLE (Li et al., 2024a) and Speculative Streaming (Bhendawade et al., 2024) recognized that sequential dependencies matter for prediction accuracy and introduced recurrence into the draft model. However, the paper identifies a complementary concern:

> "However, the reduced parallelism due to recurrence leads to lower GPU utilization, introducing overhead that diminishes speedup gains, even when the acceptance rate is substantially higher."

This is the crux of the tension: sequential (recurrent) prediction is more *accurate* because it respects token dependencies, but it's less *parallelizable* because you can't predict all future tokens simultaneously. Purely parallel prediction (Medusa) is faster per draft step but less accurate, leading to rejections. Neither extreme fully solves the problem.

### The Unsolved Challenge: Converting Compute into Speedup Efficiently

The paper positions itself squarely in this tension. The challenge, as they frame it, is to design a draft model that achieves **high predictive accuracy** (to maximize accepted tokens per LLM call) while **making effective use of the GPU's computational capacity** (to keep the draft model's overhead low enough that the net result is a speedup, not a slowdown). This is fundamentally an empirical question that depends on hardware characteristics — on a server-grade GPU with massive parallel compute, you can afford more overhead if it yields proportionally higher accuracy; on a mobile GPU with limited compute, the overhead budget is much tighter.

Section 2 explicitly frames ReDrafter's contribution in these terms:

> "ReDrafter allocates computational resources to beam search, resulting in a higher acceptance rate. The intensity of the beam search is controlled by the beam width and length, which can be adjusted based on hardware capabilities and specific implementations."

The key insight here is that beam search provides a tunable knob: you can spend more compute (wider beam) to explore more candidate sequences and increase the probability that the LLM accepts many tokens, or spend less compute (narrower beam) when hardware is constrained. This is not a fixed algorithm but a parameterized strategy that can be adapted to the deployment environment.

### Prior Work on Training Draft Models

The paper also identifies a gap in how draft models are trained. The natural approach — train the draft model to predict ground-truth next tokens from a corpus — is suboptimal because the draft model's goal isn't to predict the "correct" next token per se, but to predict what the **target LLM** would predict. These can diverge: given the same context, two different language models may assign different probabilities to plausible continuations. A draft model trained on ground-truth data may learn patterns that the LLM doesn't share, leading to rejections even when the draft's predictions are linguistically reasonable.

The solution is **knowledge distillation** (Kim & Rush, 2016; Zhou et al., 2023): train the draft model to mimic the LLM's output distribution rather than the ground-truth distribution. Prior work like Medusa2 (Cai et al., 2024) and DistillSpec (Zhou et al., 2023) had already explored this, but ReDrafter argues that the combination of (a) an RNN draft model conditioned on LLM hidden states, (b) beam search for candidate generation, and (c) distillation-based training creates a synergy that none of the prior approaches achieve individually.

The paper makes this explicit in framing ReDrafter's training:

> "ReDrafter applies knowledge distillation from LLMs, improving inference time efficiency by investing more resource in training time."

This reframes the problem: instead of trying to minimize the draft model's training cost, invest heavily in training (distillation data generation, alignment optimization) so that the inference-time draft model is maximally efficient. The training-time computation is amortized across all future inference, making it a worthwhile investment.

### The On-Device Deployment Gap

A subtle but important motivation in the paper is the growing importance of on-device LLM inference. The paper devotes significant experimental attention (Section 4.2, Appendix A.2) to Apple Silicon deployment, arguing that on-device scenarios present a unique opportunity for speculative decoding:

> "While it's well-known that on-device GPUs have less computational power and bandwidth compared to CUDA-based systems, the on-device scenario is simpler, with a single user interacting with a locally deployed LLM. This setup provides an opportunity to harness available computational resources for speculative decoding."

At batch size 1 — which is the norm for on-device assistants — the memory-bandwidth bottleneck is most severe, and the GPU has parallel compute capacity sitting idle. A draft model that can productively use that idle capacity, without overwhelming the GPU's limited resources, is exactly what on-device deployment needs. Prior draft model designs were primarily optimized for server GPUs with abundant compute; ReDrafter's tunable beam width is explicitly designed to work across both regimes.

### How ReDrafter Positions Itself

The paper's positioning is not "we invented speculative decoding" or "we invented RNN draft models" — both ideas existed. Rather, ReDrafter's contribution is in the **specific synthesis of design choices** that together overcome the limitations of prior approaches:

1. **RNN draft model conditioned on LLM hidden states** (not independent heads, not a detached model) captures sequential dependencies while leveraging the LLM's rich representations — addressing both the accuracy limitation of Medusa and the alignment problem of detached drafters.

2. **Beam search with dynamic tree attention** allows the draft model to explore multiple candidate sequences efficiently, converting additional compute into higher acceptance rates, with a tunable beam width that adapts to hardware — addressing the compute-efficiency tension identified in prior recurrent drafters.

3. **Knowledge distillation training** aligns the draft model's predictions with the target LLM rather than with ground truth — addressing the training objective mismatch.

The paper explicitly claims this synthesis yields state-of-the-art performance:

> "Our empirical results reveal that ReDrafter utilizes compute more effectively compared to previous methods, delivering state-of-the-art speedups across various implementations and hardware platforms."

This is a claim about **compute efficiency** — not just absolute speedup, but speedup per unit of additional computation invested in the draft model. The tunable beam width means ReDrafter can operate near the Pareto frontier of speedup vs. draft model cost, while prior methods (with fixed draft model architectures) may be suboptimal for particular hardware configurations.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

**What the system is:** ReDrafter is a speculative decoding system that attaches a small recurrent neural network directly to a large language model's hidden states, enabling the combined system to generate multiple tokens per LLM forward pass instead of one.

**What problem it solves and the "shape" of the solution:** The system addresses the memory-bandwidth bottleneck in autoregressive LLM inference by shifting work from memory-bound LLM calls to compute-bound draft model operations. The solution has a "predict-then-verify" shape: the draft model generates multiple candidate token sequences using beam search guided by the LLM's internal representations, the candidates are deduplicated and structured into a tree using a GPU-friendly algorithm, and the LLM verifies them all in one forward pass — accepting the longest correct prefix. This converts idle GPU compute into reduced LLM calls, producing wall-clock speedup.

### 3.2 Big-Picture Architecture (Diagram in Words)

ReDrafter has five major components operating in a loop during inference:

1. **The frozen LLM** — produces one guaranteed next token, its last-layer hidden state, and later verifies draft candidates. This is the "truth" model whose output distribution the system must match exactly.
2. **The RNN draft model** — takes the LLM's last hidden state plus the embedding of the LLM-generated token, recurrently predicts future tokens, and produces log-probabilities that drive beam search. Parameters are shared across prediction steps.
3. **Beam search module** — uses the draft model's predictions to explore multiple candidate sequences in parallel, maintaining a beam of `beam_width` candidate sequences each of length `beam_length`. This runs on GPU using the draft model's forward passes.
4. **Dynamic tree attention module** — takes the beam search output (which contains duplicated prefixes across candidates), identifies shared prefixes using GPU-parallel tensor operations, compresses the candidates into a "packed beam" without duplication, and constructs an attention mask encoding the tree structure of token dependencies.
5. **LLM verification and selection module** — runs a single forward pass of the LLM over the packed beam tokens with the tree-structured attention mask, computes log-probabilities for each draft token, selects the longest prefix matching the LLM's greedy prediction (or rejection sampling), and appends accepted tokens to the output sequence. The LLM's hidden state from this verification step feeds back into the draft model for the next iteration.

**Information flow per generation step:** Previously generated tokens + LLM's last hidden state → draft model RNN produces beam search predictions → beam search outputs candidate sequences → dynamic tree attention deduplicates into packed beam + attention mask → LLM forward pass verifies all tokens → accepted tokens appended to output → LLM's new hidden state fed to next draft model call → repeat until stopping criteria met.

### 3.3 Roadmap for the Deep Dive

- **First, the draft model architecture and its forward pass** — the core predictive component that replaces multiple independent heads with one shared recurrent structure, because understanding how the RNN conditions on LLM states and its own previous predictions is foundational to everything else.
- **Second, beam search during inference** — how the draft model's per-step probabilities are used to explore multiple candidate sequences in parallel, and why beam search is the mechanism that converts compute into acceptance rate, because this is the "compute allocation knob" that distinguishes ReDrafter from prior work.
- **Third, dynamic tree attention** — the GPU-friendly algorithm for deduplicating shared prefixes across beam search candidates, because this is a novel algorithmic contribution that makes beam search practical for speculative decoding without wasting LLM forward-pass computation on redundant tokens.
- **Fourth, the full speculative decoding loop** — how all components interact in an inference step, what gets passed where, and what selection procedures determine accepted tokens, because this connects the individual mechanisms into the complete system.
- **Fifth, training via knowledge distillation** — the training objective, why it differs from standard next-token prediction, and how the distillation data is generated, because proper training is what makes the draft model align with the LLM's behavior rather than with ground truth.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems design paper** whose core idea is that an RNN-based draft model with beam search, trained via knowledge distillation, and combined with a GPU-friendly tree attention deduplication algorithm, achieves higher wall-clock speedup than prior speculative decoding approaches because it (a) captures sequential dependencies for higher prediction accuracy, (b) converts idle GPU compute into exploration of multiple candidate sequences, and (c) avoids wasting LLM verification compute on duplicated prefixes.

---

#### Draft Model Architecture and Forward Pass

The draft model is a small recurrent neural network that predicts future tokens by conditioning on two information sources: the frozen LLM's **last-layer hidden state** (which provides a rich, contextualized representation of the entire input sequence so far) and **its own previously predicted token embeddings** (which capture local sequential dependencies among the tokens it is drafting).

**Why condition on LLM hidden states rather than using a standalone model.** As explained in Section 2, detached draft models must independently build up a representation of the context from scratch — they need to embed the input tokens, run attention, and model the probability distribution over the next token using only the surface-level token sequence. By contrast, ReDrafter's draft model receives the LLM's last hidden state `$h$`, which already encodes the entire context through the LLM's full transformer stack (multi-head self-attention over all preceding tokens, feedforward layers, etc.). This means the draft model doesn't need to re-learn context representations — it can be a lightweight predictor that "reads off" the LLM's understanding and focuses purely on modeling local token-to-token transitions. In neural network terms, this is **feature sharing**: the expensive computation of understanding context is done once by the LLM, and the draft model benefits from it at essentially zero additional cost.

**RNN state initialization.** Formally, given that the LLM has just generated a token `$x$` with embedding vector `$e_1$` (the standard learned embedding for that token) and produced last-layer hidden state `$h$`, the draft model initializes its own hidden state as a concatenation:

$$g_1 = [s_1, h]$$

where `$s_1 := e_1$` is the embedding of the token the LLM just produced, `$h$` is the LLM's last-layer output vector, and `$[\cdot, \cdot]$` denotes concatenation.

**What this initializes:** the draft model's first hidden state `$g_1$` carries both the semantic representation of the most recent token (via `$s_1$`) and the full contextual representation of the entire prefix (via `$h$`). The concatenation means these two information sources are kept separate, with the draft model's subsequent layers learning how to integrate them.

**Why concatenate rather than add or use other fusion:** concatenation preserves information from both sources without assuming they live in compatible spaces or that simple addition is the right fusion operation. The LLM's hidden state `$h$` and the token embedding `$s_1$` come from different parts of the model with different dimensionalities and semantic interpretations; concatenation lets the downstream MLP layers learn the optimal interaction, rather than imposing a hard-coded fusion (like element-wise addition) that might lose information.

**Recurrent state update (predicting the `$t$`-th future token).** For each subsequent prediction step `$t \geq 2$`, the draft model updates its hidden state using a standard RNN formulation:

$$s_t = f(U s_{t-1} + W e_t + b)$$

$$g_t = [s_t, h]$$

where `$U$` and `$W$` are learned weight matrices, `$b$` is a learned bias vector, `$e_t$` is the embedding of the token predicted at the previous draft step, and `$f$` is the activation function.

**What this computes, step by step:**

1. The previous RNN state `$s_{t-1}$` is linearly transformed by weight matrix `$U$` and added to the linear transformation of the new token embedding `$e_t$` by weight matrix `$W$`, plus bias `$b$`.
2. The activation function `$f$` (almost certainly `$\tanh$` or ReLU, though the paper doesn't explicitly specify — standard for the RNN variant they cite from Mikolov & Zweig, 2012) introduces non-linearity, producing the new RNN state `$s_t$`.
3. This new state is concatenated with the frozen LLM hidden state `$h$` (which remains constant across all draft steps), producing `$g_t$` — the draft model's hidden representation at step `$t$`.

**Why this RNN formulation rather than a transformer or MLP-only approach.** The paper explicitly states they "opt for a simple recurrent design to model the connections among the shared draft heads, deferring more complex model choices to future investigations." The key word is **shared**: unlike Medusa, which uses `$T$` separate draft heads with `$T$` separate parameter sets (one per future position), the RNN uses the **same parameters** `$U$`, `$W$`, `$b$` at every prediction step. This shared parameterization has two advantages:

1. **Parameter efficiency:** the draft model's size is independent of `$T$` (the number of future tokens being predicted). Training to predict 5 tokens uses the same number of parameters as predicting 10 tokens — only the unrolling during training changes. This keeps the draft model lightweight regardless of how many tokens it's asked to draft.
2. **Inductive bias for sequential structure:** by sharing parameters across steps, the model is forced to learn a general "given what I predicted last and the full context, what comes next" function, rather than position-specific predictors. This captures the Markov property of language — the transition dynamics at position `$t+2$` should be similar to position `$t+3$`, since both are conditioned on everything before them.

**From hidden state to token prediction.** The draft model passes `$g_t$` through "a few layers of MLPs with skip connections, followed by a standard softmax layer at the end" to produce a probability distribution over the vocabulary for the `$t$`-th future token. The skip connections (residual connections, as in ResNet architectures) help gradient flow during training and prevent degradation as more MLP layers are added. The softmax normalizes the output logits into a proper probability distribution.

**What "a few layers of MLPs" means concretely.** The paper does not specify the exact MLP architecture (number of layers, hidden dimension, activation function between layers), which is a notable detail gap. Based on the context — this is a lightweight draft model that must run faster than a full LLM forward pass — the MLP is likely 2–4 layers with moderate hidden dimensions (perhaps 1024–4096, depending on the base LLM size), using GeLU or ReLU activations. The skip connections likely operate at the MLP layer granularity: the output of MLP layer `$i$` is added to its input before being fed to layer `$i+1$`.

**Putting it together — a concrete forward pass example.** Suppose the LLM has generated "She enjoyed the quiet" and produced hidden state `$h$`. The draft model:
1. Takes `$s_1 = \text{embed}(\text{"quiet"})$` and `$h$`, produces `$g_1 = [s_1, h]$`, feeds through MLP+softmax → predicts "morning" (token for step 2).
2. Sets `$e_2 = \text{embed}(\text{"morning"})$`, updates RNN: `$s_2 = f(U s_1 + W e_2 + b)$`, concatenates with `$h$`: `$g_2 = [s_2, h]$`, feeds through MLP+softmax → predicts "sipping" (token for step 3).
3. Sets `$e_3 = \text{embed}(\text{"sipping"})$`, updates RNN: `$s_3 = f(U s_2 + W e_3 + b)$`, concatenates with `$h$`: `$g_3 = [s_3, h]$`, feeds through MLP+softmax → predicts "coffee" (token for step 4).
4. And so on for `$T$` steps (where `$T$` is the beam length).

**Key distinction from Medusa.** In Medusa, predicting "morning" at position 2 and "sipping" at position 3 would use completely separate parameter sets — head 2 doesn't know head 1 predicted "morning," and head 3 doesn't know head 2 predicted "sipping." In ReDrafter, the prediction of "sipping" explicitly conditions on the fact that "morning" was predicted, because the RNN state `$s_3$` incorporates the embedding of "morning" through the recurrence. This is what "leveraging the sequential structure" means — the draft model builds a coherent chain of tokens rather than predicting each position in isolation.

**Why this matters for the beam search (preview).** Because the draft model is sequential, to explore different possible continuations at step `$t$`, you need to maintain multiple RNN states — one per beam candidate — each conditioned on different histories of previously predicted tokens. This is exactly what beam search does, and it's why the RNN architecture and beam search are complementary: the RNN provides the sequential modeling that makes predictions accurate, and beam search provides the parallelism that makes exploration computationally feasible.

---

#### Beam Search During Inference

Beam search is the mechanism that converts the draft model's per-step probability distributions into a set of high-probability candidate sequences that the LLM will later verify. Rather than greedily taking the single most probable token at each step (which gives only one candidate sequence) or sampling randomly (which may miss good candidates), beam search systematically explores the `$K$` most promising partial sequences at each step, where `$K$` is the beam width.

**Why beam search rather than greedy or sampling for speculative decoding.** The goal of speculative decoding is to maximize the number of tokens the LLM accepts per verification step. To achieve this, the set of candidate sequences sent to the LLM should maximize the probability that **at least one** candidate has a long prefix matching what the LLM would have generated. Greedy decoding produces exactly one sequence; if it's wrong at step 2, you get zero additional tokens accepted. Random sampling with temperature can explore, but without systematic pruning, it may waste probability mass on low-likelihood sequences. Beam search strikes a balance: it explores diverse possibilities while concentrating computation on high-probability regions, increasing the chance that one beam contains the LLM's preferred continuation.

**The beam search procedure step by step:**

1. **Initialization:** The draft model computes the probability distribution over the vocabulary for the first future token (position `$t+1$` when the LLM just generated token `$t$`). This requires one forward pass of the RNN draft model, producing `$P(y_{t+1} | \text{context})$`. The top `$K$` tokens by probability are selected as the first tokens of `$K$` candidate beams. Each beam maintains its own RNN hidden state `$s$` (since different first tokens lead to different recurrent states).

2. **Expansion (step `$i \in \{2, \ldots, T\}$`):** For each of the `$K$` active beams, the draft model computes the next-token distribution conditioned on that beam's history. Specifically, for beam `$j$` with current state `$s^{(j)}$` and most recent token `$e^{(j)}$`, update via `$s^{(j)} = f(U s^{(j)} + W e^{(j)} + b)$`, produce logits via MLP+softmax, obtaining `$P(y^{(j)}_{t+i} | \text{beam } j \text{'s history})$`. This requires `$K$` parallel forward passes of the draft model (or equivalently, one batched forward pass with batch size `$K$`).

3. **Scoring and pruning:** Each of the `$K \times |V|$` possible extensions (every beam extended by every possible next token) receives a score equal to the cumulative log-probability of the sequence so far. The top `$K$` scoring extensions overall are selected to form the `$K$` beams for the next step. The beams that don't make the cut are discarded.

4. **Repeat:** Steps 2–3 repeat for `$T$` total steps (where `$T$` is the beam length). At the end, there are `$K$` candidate sequences, each of length `$T$`.

**What beam width `$K$` and beam length `$T$` control:**
- **Beam width `$K$`:** the number of candidate sequences explored in parallel. Larger `$K$` means more exploration, higher probability of finding a sequence the LLM will accept, but proportionally more draft model computation and more tokens for the LLM to verify.
- **Beam length `$T$`:** the number of future tokens drafted per sequence. Larger `$T$` means potentially more tokens accepted per LLM call, but predictions become less accurate at later positions (since the draft model's errors compound) and the draft model must be trained to predict at least `$T$` tokens ahead.

**The computational cost of beam search.** With beam width `$K$` and beam length `$T$`, the total number of draft model forward passes is `$K \times T$` (one per beam per step). However, these can be **batched**: at each beam search step, all `$K$` beams' RNN updates and MLP predictions can run in parallel on the GPU as a single batch of size `$K$`. This is the critical insight that makes beam search practical on GPUs — the additional computation from exploring `$K$` paths is parallelized, so the wall-clock overhead is much less than `$K \times$` the single-sequence cost. The paper explicitly notes this:

> "While a wider beam requires more FLOPs for beam search and for LLM verification, a powerful GPU can process these FLOPs in parallel, minimizing the increase in wall time."

**The tradeoff between beam width and GPU capability.** This explains a key experimental finding previewed in Section 3.2 and detailed in Section 4. On a powerful server GPU (Nvidia H100), the optimal beam width is large (50+ by PyTorch experiments) because the GPU has abundant parallel compute to absorb the additional FLOPs without significant wall-clock increase. On a mobile GPU (Apple M1 Max), the optimal beam width is only 1 — the GPU is compute-limited, and even the modest overhead of beam width 2 slows down inference more than the extra accepted tokens compensate. The M2 Ultra, with more GPU compute, achieves optimal speedup at beam width 3. This hardware-dependence is not a bug but a feature: the beam width is a tunable parameter that can be adjusted per deployment, making ReDrafter adaptable to different hardware without architectural changes.

**How beam search interacts with the RNN draft model.** Because the RNN is sequential — predicting step `$i$` requires knowing the token predicted at step `$i-1$` — beam search must maintain `$K$` separate RNN states throughout the search. At each step, each beam independently updates its state based on its own history. This is different from Medusa, where each position's prediction is independent and there are no per-beam states to maintain. The additional state maintenance is a cost of the sequential modeling, but the paper argues (and the speedup results demonstrate) that the improved accuracy from sequential conditioning more than compensates for this overhead.

**Training implications for beam length.** The draft model is trained to predict `$T_{\text{train}}$` tokens ahead. At inference, the beam length can be set to any value — the RNN parameter sharing means it generalizes to lengths seen during training or (with some degradation) beyond. The paper trains with a prediction range of 5 tokens and finds the optimal beam length during inference is around 4–5 (Appendix A.2, Figure 7), which matches the training horizon. Setting beam length longer than the training horizon degrades accuracy because the model hasn't learned to plan that far ahead; setting it shorter leaves potential speedup on the table because the draft model's later-step predictions could have been accepted.

---

#### Dynamic Tree Attention

Dynamic tree attention is a GPU-friendly algorithm for eliminating redundant computation in the LLM verification step. The problem it solves is straightforward: when beam search produces `$K$` candidate sequences, many of them share common prefixes because they were derived from the same high-probability beams at early steps. For instance, in Figure 4(a), three candidate sequences are:
- "morning sipping coffee and reading"
- "morning sipping coffee and watching"  
- "morning sipping coffee on her"

All three share the prefix "morning sipping coffee". If we naïvely concatenated all three sequences and sent them to the LLM for verification, we would compute the LLM's forward pass on the tokens "morning", "sipping", and "coffee" three separate times — once per candidate — even though the LLM's prediction for these tokens is identical regardless of which candidate they belong to. This redundant computation wastes the LLM's forward-pass capacity on tokens that don't provide new information.

**What dynamic tree attention does.** It identifies shared prefixes across all beam search candidates, removes duplicate tokens, arranges the remaining unique tokens into a tree structure, and constructs a custom attention mask that ensures each token attends only to its valid ancestors in the tree (not to tokens from different tree branches that happen to be adjacent in the flat representation). The LLM then processes this "packed beam" in a single forward pass, computing predictions for each unique token exactly once.

**The difference from static tree attention in prior work.** Prior work (Miao et al., 2023; Spector & Re, 2023; Cai et al., 2024) also used tree structures to deduplicate shared prefixes among candidate sequences. However, in those approaches, the tree structure was **determined at design time** — for example, Medusa knows in advance which heads predict which positions and can pre-define a fixed tree pattern that covers likely token combinations. ReDrafter's tree structure is **dynamic**: it depends on the actual beam search results at runtime, which vary from step to step depending on the LLM's hidden state and the draft model's predictions. You cannot pre-define a fixed tree pattern because you don't know in advance which sequences will share prefixes — one beam search step might produce candidates that diverge at step 2, while the next might produce candidates that share prefixes through step 4.

**The algorithmic challenge.** Building a dynamic tree from beam search results is conceptually simple with a trie data structure: insert each candidate sequence into the trie, merging shared prefixes, and collect the unique nodes. However, a standard trie implementation involves sequential pointer-chasing (insert one token, follow the pointer to the child node, insert the next token, etc.), which is fundamentally sequential and cannot be parallelized across GPU threads. For beam widths of 50–70 and beam lengths of 5, this would become a serial bottleneck, undermining the speedup from speculative decoding.

**The key insight for GPU-friendliness.** The paper's critical observation is:

> "We notice that a unique property of our problem is that all candidate sequences have the same length."

Because beam search always produces `$K$` sequences each of exactly `$T$` tokens, all candidates are the same shape — a `$(K, T)$` tensor. This rectangular structure means that prefix-matching operations can be expressed as tensor operations operating on the entire beam at once, rather than sequential per-sequence operations.

**The `dedup_prefix` algorithm (Appendix A.1, Listing 1).** The algorithm takes a beam tensor of shape `$(\text{beam\_width}, \text{beam\_length})$` and produces a `prefix_tree` tensor of the same shape where `prefix_tree[i][j] = k` means "candidate sequence `$i$` shares its prefix of length `$j+1$` with candidate sequence `$k$` (the smallest index that shares this prefix)." The implementation, described as "five lines of code," works as follows:

1. **Build a 3D boolean `matches` tensor:** `matches[i][j][k]` is `True` if the `$k$`-th token of candidate `$i$` equals the `$k$`-th token of candidate `$j$`. This is computed via broadcasting: `beam[:, :, None] == beam[:, None, :]`, which compares every token in every candidate against every token in every candidate, producing a `$(\text{beam\_width}, \text{beam\_width}, \text{beam\_length})$` boolean tensor.

2. **Accumulate matches to find shared prefixes:** For two sequences to share a prefix of length `$p$`, they must match at positions 0, 1, ..., `$p-1$`. The cumulative sum `torch.cumsum(matches, dim=2)` counts, for each pair of sequences, how many tokens match up to each position. This is compared against a target tensor `[1, 2, 3, ..., beam_length]` (incremented integers) — sequences `$i$` and `$j$` share a prefix of length `$p$` if their cumulative match count at position `$p-1$` equals `$p$` (meaning they matched at every position up to `$p-1$`).

3. **Find the smallest index that shares each prefix:** `torch.argmax(seq_matches, dim=2)` returns, for each candidate `$i$` and each prefix length, the smallest candidate index `$j$` that shares that prefix. This is the `prefix_tree` output.

**What this produces concretely.** For the example beam:
```
beam = [[quiet, morning, sipping, coffee, and],
        [quiet, morning, sipping, coffee, reading],
        [quiet, morning, sipping, on,     her]]
```
The `prefix_tree` would indicate that candidate 1 shares its prefix of length 4 ("quiet morning sipping coffee") with candidate 0, and candidate 2 shares its prefix of length 3 ("quiet morning sipping") with candidate 0. Tokens where `prefix_tree[i][j] < i` can be deduplicated — candidate `$i$`'s token at position `$j$` is already represented in candidate `prefix_tree[i][j]`.

**The `pack_beam` function.** After identifying shared prefixes, `pack_beam` compresses the beam into a "packed beam" — a flat sequence of unique tokens, with no duplication. The attention mask is then constructed to reflect the tree structure: each token in the packed beam attends to all its ancestors (tokens in the shared prefix leading to it) but not to "sibling" tokens from other branches or tokens that come after it in the flat representation. This ensures the LLM's self-attention operates correctly — a token like "reading" should attend to "quiet morning sipping coffee" but not to "watching" or "her", which are from different continuations.

**Compression ratio.** The empirical results in Figure 6 (left) show that dynamic tree attention reduces the number of tokens the LLM must process by 30–60%, with a "consistent compression ratio across different beam sizes." For example, with beam width 45 (225 total tokens), the packed beam contains approximately 90–130 tokens. This reduction directly translates to less LLM forward-pass computation, making the verification step faster for the same number of beam candidates. The compression ratio is stable across the 1st, 50th, and 99th percentiles, meaning it's predictable and reliable, not subject to pathological cases.

**Why this matters for the speedup equation.** Without dynamic tree attention, wider beams increase the LLM verification cost linearly — beam width 50 means processing 250 tokens per verification step (with beam length 5), which might take longer than the time saved by accepting more tokens. With dynamic tree attention, the verification cost scales sub-linearly with beam width because tokens are shared across candidates. This is what makes large beam widths (50+, as used in the PyTorch experiments) practical: the LLM verification step doesn't process `$K \times T$` tokens, but rather `$(K \times T) \times (1 - \text{compression\_ratio})$` unique tokens.

**Generality of dynamic tree attention.** The paper notes this technique is not specific to ReDrafter:

> "The use of dynamic tree attention is not limited to ReDrafter. It can also be used in detached speculative decoding approaches while a separate draft model performs beam search and then apply dynamic tree attention."

Any speculative decoding setup where the draft model produces multiple candidate sequences with shared prefixes (which is essentially all beam-search-based approaches) can benefit from this GPU-friendly deduplication.

---

#### The Full Speculative Decoding Loop

Now that we understand each component individually, we can describe the complete inference loop that ties them together. The paper describes this in Section 3.4, and the process is illustrated in Figure 2.

**Step 1: LLM generates the guaranteed first token.** At the start of each generation step, the LLM takes the full context (all previously generated tokens) and performs a standard autoregressive forward pass, producing:
- One new token (the "initial token" for this step, highlighted in green in Figure 2)
- The last-layer hidden state `$h$` corresponding to this token

This initial token is **guaranteed** — it is the LLM's own prediction and will always be part of the final output. ReDrafter does not attempt to draft this token; it starts drafting from the next position.

**Step 2: Draft model runs beam search.** Using the LLM's hidden state `$h$` and the embedding of the initial token `$e_1$`, the RNN draft model performs beam search as described in Section 3.4.2, producing `$K$` candidate sequences each of length `$T$`. These `$K \times T$` tokens represent the draft model's best guesses for what the LLM would generate if run autoregressively for `$T$` more steps.

**Step 3: Dynamic tree attention compresses the beam.** The `dedup_prefix` and `pack_beam` functions process the `$(K, T)$` beam tensor, producing:
- A packed beam: a flat sequence of unique tokens (no duplicated prefixes)
- A tree-structured attention mask encoding which tokens can attend to which others

The packed beam may contain significantly fewer tokens than `$K \times T$`, depending on how many shared prefixes the beam search discovered (typically 30–60% reduction, per Figure 6).

**Step 4: LLM verifies the packed beam.** The LLM runs a single forward pass over the packed beam tokens with the tree attention mask. This forward pass computes:
- The LLM's log-probability for each draft token under the LLM's own distribution
- A new hidden state `$h'$` corresponding to the last token in the packed beam (which will feed into the next draft model call)

The verification is **exact**: the tree attention mask ensures that each draft token attends only to its valid prefix (the tokens that precede it in the tree), so the LLM's prediction for each token is identical to what it would be if those tokens had been generated autoregressively. This is guaranteed because the self-attention mechanism in transformers respects the attention mask — tokens that are masked out contribute zero to the attention computation.

**Step 5: Selection — find the longest accepted prefix.** The verification step produces LLM log-probabilities for every draft token. The system must now decide which tokens to accept. The paper mentions two selection methods:

> "The selection method can range from a greedy approach (aka. token matches), to rejection sampling."

**Greedy matching (temperature = 0):** Each draft token is compared against the LLM's greedy prediction at that position (the token that maximizes the LLM's log-probability). The system scans through the drafted sequences in order of beam scores, finding the candidate with the longest prefix where every token matches the LLM's greedy choice. All tokens in that prefix are accepted and appended to the output sequence.

**Rejection sampling (temperature > 0):** For non-greedy decoding, the standard speculative decoding rejection sampling procedure from Leviathan et al. (2023) is used: for each draft token in sequence, generate a random number and accept the token with probability proportional to `$p_{\text{LLM}}(x) / p_{\text{draft}}(x)$`. If a token is rejected, all subsequent tokens in that candidate are also rejected. This guarantees the output distribution matches the LLM's distribution exactly, even with stochastic sampling.

**What "longest accepted prefix" means concretely.** With multiple candidate sequences from beam search, you don't just take the first candidate's accepted tokens. You compare across all candidates and select the one whose accepted prefix is longest. For example, if candidate 1 has 3 tokens accepted ("morning sipping coffee"), candidate 2 has 4 tokens accepted ("morning sipping coffee and"), and candidate 3 has 2 tokens accepted ("morning on"), you accept 4 tokens from candidate 2. This is the advantage of beam search over generating a single draft sequence — you get to pick the best among `$K$` attempts.

**Step 6: Update context and iterate.** The accepted tokens are appended to the previously generated sequence. The LLM's hidden state `$h'$` from the verification forward pass (Step 4) and the embedding of the last token before the next position to predict become the inputs for the draft model in the next generation step. The process repeats from Step 1 until the stopping criteria are met (end-of-sequence token generated or maximum length reached).

**Why the hidden state from verification feeds back.** The LLM's forward pass during verification already computes hidden states for every token position in the packed beam. By extracting the hidden state corresponding to the final accepted position, ReDrafter avoids running a separate LLM forward pass just to produce the hidden state for the draft model — it reuses computation that was already done for verification. This is an efficiency detail: in the next step's "Step 1," the LLM's forward pass to produce the guaranteed next token starts from the last accepted token's position, using the cached hidden state from the verification step.

**Correctness guarantee.** The paper emphasizes:

> "ReDrafter guarantees the generated sequence matches LLM's output."

This is a crucial property inherited from speculative decoding: because the LLM verifies every draft token against its own distribution (either greedily or via rejection sampling), the output is identical to what the LLM would have produced autoregressively. ReDrafter changes *how* tokens are generated (draft-then-verify rather than one-at-a-time), not *what* tokens are generated. This is not an approximate method — it's an exact acceleration technique with no quality degradation.

**Computational tradeoff summary per step.** In one ReDrafter generation step:
- **Computation done:** 1 LLM forward pass (for the initial token) + `$K \times T$` RNN draft model forward passes (for beam search, batched into `$T$` batches of size `$K$`) + 1 LLM forward pass (for verification over the packed beam)
- **Tokens generated:** 1 (guaranteed initial token) + accepted_prefix_length (typically 2–4 tokens, as shown by Tokens/Step in Table 1)
- **Net effect:** 2 LLM forward passes produce 3–5 tokens, versus 3–5 LLM forward passes in autoregressive decoding. The savings come from eliminating the middle `$(T-1)$` LLM calls.

The speedup depends on three factors: (a) how long the accepted prefix is (draft model accuracy), (b) how much overhead the beam search adds (draft model efficiency), and (c) how many redundant tokens dynamic tree attention eliminates (compression ratio). These three factors correspond exactly to ReDrafter's three design contributions — the RNN draft model improves (a), beam search balances (a) and (b), and dynamic tree attention improves (c).

---

#### Training via Knowledge Distillation

The draft model's training objective is fundamentally different from standard language model training. A standard language model is trained to maximize the probability of the ground-truth next token given the context — it learns to predict what actually appears in the training data. But the draft model for speculative decoding doesn't need to predict ground truth; it needs to predict **what the target LLM would predict**, because the LLM's acceptance decision depends on whether the draft token matches the LLM's own output, not whether it matches ground truth.

**The distributional mismatch problem concretely.** Suppose the context is "She enjoyed the quiet morning, sipping coffee and" and the ground truth continuation is "reading a novel." The LLM might assign 60% probability to "reading", 25% to "watching", and 15% to other tokens. A draft model trained on ground truth would be optimized to output "reading" with high confidence. This is correct for the ground truth distribution but suboptimal for the speculative decoding interaction: if the draft model knows that the LLM assigns non-trivial probability to "watching", it might be better to include "watching" as a beam candidate (since sometimes the LLM will sample it), even though it's not the ground-truth token.

More subtly, if the LLM has systematic biases — it overuses certain phrasings or makes consistent errors on certain constructions — a ground-truth-trained draft model would predict the "correct" token while the LLM would generate the "wrong" one, leading to rejections. A distillation-trained draft model learns to mimic the LLM's idiosyncrasies.

**The KL divergence objective (Equation 1).** The paper formalizes this as minimizing the KL divergence between the LLM's predictive distribution and the draft model's distribution:

$$\min_{p_{\text{draft}}} KL(p_{\text{llm}}(y_{1:T}) \| p_{\text{draft}}(y_{1:T})) = \min_{p_{\text{draft}}} \mathbb{E}_{p_{\text{llm}}(y_{1:T})} [-\log p_{\text{draft}}(y_{1:T})]$$

where `$p_{\text{llm}}(y_{1:T})$` is the LLM's joint distribution over a sequence of `$T$` tokens (conditioned on the context, which is omitted for brevity) and `$p_{\text{draft}}(y_{1:T})$` is the draft model's joint distribution over the same sequence.

**What this equation computes:** the KL divergence measures how many extra bits (or nats) are needed to encode samples from the LLM's distribution using the draft model's distribution instead of the LLM's own distribution. Minimizing this makes the draft model's predictions as similar as possible to the LLM's, in the sense of information-theoretic distance. The expectation is taken over sequences drawn from the LLM, meaning we want the draft model to be accurate specifically on sequences the LLM is likely to generate, not on all possible sequences.

**Why KL divergence rather than cross-entropy with ground truth:** standard training minimizes `$-\log p_{\text{draft}}(y_{\text{gt}})$` where `$y_{\text{gt}}$` is the ground-truth token. This is equivalent to minimizing KL divergence to a delta distribution at the ground-truth token. But the LLM's distribution is not a delta — it's a full probability distribution over the vocabulary. KL divergence to the LLM's distribution penalizes the draft model for placing low probability on tokens the LLM finds plausible, even if they're not the single ground-truth token. This encourages the draft model to spread probability mass similarly to the LLM, which is exactly what's needed for beam search: the beams should explore tokens the LLM considers likely, not just the single most common token in the training data.

**The empirical loss (Equation 2).** Since we can't directly minimize the expectation over the LLM's distribution (it's intractable to compute exactly), the paper uses a sampling-based approximation:

$$\min_{p_{\text{draft}}} \mathcal{L}_{\text{distill}} = \min_{p_{\text{draft}}} \sum_{t} -\log p_{\text{draft}}(\hat{y}_{t+1:t+T} | y_{1:t})$$

where `$\hat{y}_{t+1:t+T}$` is a sequence of `$T$` tokens sampled from the LLM conditioned on the ground-truth context `$y_{1:t}$`, and the sum runs over all positions `$t$` in the training sequences.

**What this computes in operational terms:**

1. For each training sequence, iterate through every position `$t$`.
2. At position `$t$`, feed the ground-truth context `$y_{1:t}$` to the LLM and sample `$T$` future tokens `$\hat{y}_{t+1}, \ldots, \hat{y}_{t+T}$`. The paper uses a temperature of 0 for this sampling, meaning they take the LLM's greedy predictions — this is deterministic and produces the LLM's most likely continuation.
3. Train the draft model to predict `$\hat{y}_{t+1:t+T}$` given the context `$y_{1:t}$` as input. The loss is the negative log-likelihood of the draft model's predictions for these `$T$` tokens, summed (or averaged) over all positions.

**Why use ground-truth context but LLM-sampled targets.** The context `$y_{1:t}$` is from the ground-truth data (e.g., the training set of a dialogue dataset). This ensures the training data covers diverse, natural contexts. The targets `$\hat{y}_{t+1:t+T}$` are from the LLM, ensuring the draft model learns to mimic the LLM's behavior rather than the ground truth. This is the critical distinction: the input context is real data, but the supervision signal comes from the LLM. The paper calls this "distillation locally" because it applies at the token level: for each position, the LLM generates targets, and the draft model learns to match them.

**The "only backpropagate through the draft model" design choice.** The paper emphasizes:

> "In contrast, ReDrafter only backpropagates through the draft model, keeping the LLM unchanged to ensure the decoding results remain consistent."

This means the LLM is frozen during draft model training. The LLM is used only to generate training targets (forward passes, no gradient computation) and to provide hidden states as input features to the draft model. This has two advantages: (1) the LLM's behavior doesn't change during draft model training, so the draft model is always targeting a fixed distribution, and (2) training is much cheaper because gradients don't need to flow through the LLM's full parameter set (billions of parameters). Only the draft model's parameters (the RNN, the MLP layers, the softmax projection) are updated — typically millions of parameters, not billions.

**Training data generation procedure.** The paper describes an offline data generation process:
1. Take the training data (the paper uses dialogue datasets; Section 4 mentions MT-Bench and Alpaca for evaluation, and the training data likely comes from the same or similar sources — Vicuna's training data).
2. For each training sequence, at each position, feed the context to the frozen LLM.
3. Have the LLM generate `$T = 5$` tokens using temperature 0 (greedy decoding) — these are the distilled targets `$\hat{y}_{t+1:t+5}$`.
4. Store these context-target pairs as the training dataset for the draft model.

**Training hyperparameters.** The paper does not provide detailed training hyperparameters in the main text. It references using "a few layers of MLPs with skip connections" and a single-layer RNN, but exact architecture dimensions, learning rates, batch sizes, optimizer choices, and training durations are not specified. This is a notable omission for reproducibility. Based on the text, the training procedure likely uses standard deep learning practices: Adam or AdamW optimizer, moderate learning rates (perhaps `$10^{-3}$` to `$10^{-4}$`), and training until the draft model's predictions converge on a validation set.

**The effect of distillation on performance (Table 4).** The ablation study demonstrates that distillation training provides "an approximate 10% increase in the speedup and the average accepted tokens per step" compared to training on ground-truth tokens. For example, at beam width 64, distillation yields 2.18× speedup with 3.58 tokens per step, versus 1.99× speedup with 3.30 tokens per step without distillation. The gap is consistent across all beam widths tested (1, 2, 4, 16, 64). This validates the core premise: aligning the draft model with the LLM's distribution, rather than with ground truth, directly improves speculative decoding efficiency.

**Why not use on-policy data (the draft model's own predictions) for training.** In standard knowledge distillation for speculative decoding, one might train the draft model on the LLM's outputs, then use that draft model to generate new training data (by running the LLM on the draft model's predictions), and iterate — this is the self-improvement loop. The paper doesn't explore this, noting only that ReDrafter trains once and keeps the LLM fixed. The ReST$^{EM}$ experience from the reference paper's Appendix K (where on-policy self-improvement degraded revision model performance) suggests that offline distillation with fixed LLM targets may be more stable.

**The relationship between beam length and training horizon.** The draft model is trained to predict `$T = 5$` future tokens. At inference, the beam length is typically set to 4 or 5 (Appendix A.2, Figure 7 shows optimal speedup at beam lengths close to the training length). This makes intuitive sense: the draft model was optimized to predict up to 5 tokens ahead, so asking it to predict more than 5 tokens pushes it beyond its training distribution, where accuracy degrades. The RNN parameter sharing means it can technically predict to any length, but accuracy drops off because the model hasn't seen examples of 6th, 7th, or 10th token predictions during training and hasn't learned to maintain state quality over longer horizons.

**Summary of the training philosophy.** The paper frames distillation as "investing more resource in training time" to improve inference-time efficiency. This is an economic tradeoff: generating the distilled training data (one LLM forward pass per position per training sequence) is computationally expensive, but it's done once and amortized over all future inference. For a model that will be deployed and used thousands or millions of times, the one-time training cost is negligible compared to the cumulative inference-time savings. This is the same philosophy behind many ML deployment optimizations: move computation from inference time to training time wherever possible.

## 4. Key Insights and Innovations

### Innovation 1: Reframing Draft Model Design as a Compute-to-Accuracy Conversion Problem, with the RNN as a Parameter-Shared, Tunable Converter

The dominant framing in prior speculative decoding work treats the draft model's job as *prediction*: given the context, guess what the LLM will say next, and do it fast. This leads naturally to evaluating draft models by their standalone perplexity or top-1 accuracy. ReDrafter's deeper contribution is to reframe the draft model's job as **converting idle GPU compute into higher LLM acceptance rates** — a resource conversion problem, not just a prediction problem.

This reframing changes what you optimize. Medusa (Cai et al., 2024) adds more independent prediction heads to use more GPU parallelism, but each head operates in isolation — there's no mechanism by which spending extra compute on head 3 improves the prediction of head 4, because the heads don't share information. The paper identifies this as the root cause of diminishing returns: "independent predictions become less accurate as T increases, resulting in suboptimal predictive accuracy and a lower acceptance rate from the LLM." The compute is being spent, but it isn't being *converted* into accuracy because each head faces the same hard problem (predicting far-future tokens from a single hidden state) without help from its neighbors.

ReDrafter's RNN draft model embodies the conversion insight directly. By sharing parameters across prediction steps — the same `$U$`, `$W$`, `$b$` matrices predict token `$t+2$` as predict token `$t+3$` — every forward pass of the RNN builds on the outputs of previous passes. The compute spent predicting "morning" produces an RNN state `$s_2$` that makes predicting "sipping" easier than it would be from the LLM hidden state alone. This is the conversion mechanism: **compute is transformed into state, and state improves future predictions**. The paper shows this yields higher Tokens/Step (4.20 vs. EAGLE's 3.96 for Vicuna 7B on MT-Bench, Table 1) — meaning more of the draft computation translates into accepted tokens.

The tunable beam width is the operational lever this reframing enables. If the draft model converts compute into accuracy, the question becomes: *how much compute should you feed it?* On an H100, you feed it a lot (beam width 50+ in PyTorch experiments) because GPU parallelism absorbs the cost and the accuracy gains translate to speedup. On an M1 Max, you feed it almost nothing (optimal beam width 1, Table 2) because the GPU is compute-saturated and additional draft computation costs more wall-clock time than the extra accepted tokens save. This is not a fixed architecture choice but a deployment-time knob — a direct consequence of treating the draft model as a tunable compute-to-accuracy converter rather than a fixed-accuracy predictor. Prior work (Medusa, EAGLE) offered much less flexibility here: Medusa's number of heads is a training-time architecture decision, not a runtime dial.

This is a **fundamental reframing**, not a small refinement. It changes the design criterion from "make the most accurate draft model possible" to "make a draft model whose accuracy scales efficiently with allocated compute, and expose a knob to control that allocation." This is closer in spirit to how we think about inference-time compute in LLM reasoning (the reference paper's core contribution) than to how the speculative decoding literature had previously thought about draft models.

---

### Innovation 2: Identifying Beam Search as a Uniquely Well-Suited Exploration Strategy for RNN Draft Models, Enabling a Compute-to-Diversity Conversion That Prior Drafters Couldn't Exploit

Beam search is not a new algorithm — it's been a standard tool in sequence generation for decades. But its application to *speculative decoding draft models* represents a non-obvious design choice, and the paper's analysis reveals why it's specifically powerful when paired with an RNN drafter in a way it wouldn't be with prior approaches.

The key diagnostic: Medusa's independent heads produce an "exponentially large set of feasible candidate token sequences" (Section 1) because each head predicts independently — head 1 outputs a distribution, head 2 outputs a distribution, and the Cartesian product of their top-k choices explodes combinatorially. Medusa handles this with tree attention over a pre-defined pattern, but the fundamental problem remains: many of those combinations are incoherent (head 1 predicted "morning" and head 2 predicted "her" in a context where that makes no sense), and the verification step wastes compute on nonsensical sequences.

ReDrafter's RNN draft model doesn't have this problem because its predictions are sequential — the probability of "her" at step 3 explicitly conditions on "morning sipping" at steps 1–2, so incoherent combinations are naturally assigned low probability. But this sequential structure means you *can't* just take the top-k from each step independently — you need an algorithm that explores the joint space of sequences while respecting the sequential dependencies. Beam search is exactly that algorithm: it maintains `$K$` hypotheses, extends each by one token, scores all `$K \times |V|$` extensions, and prunes back to `$K$`. The sequential RNN predictions feed naturally into this framework, and the beam search explores diverse continuations that are all *coherent* (because they're built step-by-step using the sequential model) rather than combinatorially random.

What makes this intellectually distinctive is that beam search serves a **dual purpose** that neither prior approach could exploit. For a detached draft model, beam search would just be a way to get multiple candidates from a single model — useful, but the draft model itself doesn't change. For Medusa's independent heads, beam search over the joint space is essentially what the tree attention mechanism approximates, but the independent predictions limit how much beam search can help because the underlying predictions at later positions are already degraded. For ReDrafter's RNN, beam search simultaneously (a) explores diverse sequences to increase the chance one matches the LLM, and (b) **allows the RNN to condition on different histories**, meaning the accuracy improvement from sequential conditioning compounds with the diversity improvement from beam exploration. Beam width 1 gets you sequential accuracy but no diversity; beam width 50 gets you sequential accuracy *on 50 different paths*, maximizing the chance that one path's predictions align with the LLM.

The evidence for this synergy is in the scaling behavior. Table 1 shows ReDrafter achieving 4.20 Tokens/Step on Vicuna 7B MT-Bench versus EAGLE's 3.96 — both use recurrence, but ReDrafter's beam search extracts more accepted tokens per step. Table 3 shows that increasing beam width from 1 to 64 approximately doubles the per-request TPS (62.55 → 110.64 at batch size 1) because wider beams accept more tokens per step, and the GPU parallelizes the additional computation. This isn't just "beam search is good" — it's "beam search over an RNN drafter's sequential predictions converts GPU parallelism into acceptance rate more efficiently than independent prediction heads or fixed-tree approaches."

This is an **incremental but architecturally significant** advance: beam search itself is old, and RNN draft models existed, but the combination exploits a property (sequential predictions enable coherent beam exploration, which enables compute-to-diversity conversion) that neither component alone provides. It's the *interaction* between the RNN architecture and the beam search algorithm that's novel, not either piece in isolation. The paper makes this clear by showing that the beam width is the primary knob for trading off compute vs. acceptance rate, and that its optimal setting depends on hardware — a direct consequence of this interaction being the central mechanism for speedup.

---

### Innovation 3: Dynamic Tree Attention as a GPU-Native Algorithm That Removes the Serial Bottleneck from Beam-Based Speculative Decoding, Making Wide Beams Practical

Tree-structured attention for deduplicating shared prefixes in candidate sequences existed before ReDrafter — Miao et al. (2023), Spector & Re (2023), and Cai et al. (2024) all used it. But those approaches used **static trees** — the tree structure was pre-defined at design time, not computed from the actual draft model outputs. This works when you know the structure of the draft model's predictions in advance (e.g., Medusa knows which heads correspond to which positions), but it breaks down when the draft model's outputs are dynamic, as they are with beam search over an RNN: you cannot pre-define a tree that efficiently covers all possible beam search outcomes without being either too small (missing shared prefixes and wasting LLM compute) or too large (including tokens that won't actually appear and adding overhead).

The paper's key algorithmic insight is that **the beam search output is rectangular** — all candidates have exactly the same length — and this property enables a GPU-parallel tensor algorithm for prefix deduplication rather than a sequential trie-based approach. The `dedup_prefix` algorithm in Appendix A.1 (Listing 1) uses broadcasting and cumulative sums — operations that GPUs execute massively in parallel — to find all shared prefixes in a single pass over the beam tensor. This is not just an implementation detail; it's what makes beam search viable at scale for speculative decoding. Without it, the prefix deduplication step would be a serial bottleneck (building a trie by processing tokens one at a time) that grows with beam width, undermining the speedup from wider beams. With it, the compression step adds negligible overhead regardless of beam width.

The intellectual contribution here is recognizing that the **shape constraint** of beam search (fixed-length candidates) transforms a traditionally sequential data structure problem (trie construction) into an embarrassingly parallel tensor operation. This is a specific instance of a broader principle: when you can express a problem in terms of fixed-shape tensor operations, you unlock GPU parallelism. The paper doesn't claim this principle as novel — it's the foundation of deep learning frameworks — but the application to dynamic tree construction for speculative decoding is clever and non-obvious. Most practitioners, when faced with "deduplicate shared prefixes across variable numbers of beam candidates," would reach for a trie, not a broadcasting-based tensor comparison.

The practical impact is substantial. Figure 6 (left) shows 30–60% compression across beam widths from 5 to 70 — meaning the LLM verification step processes roughly half the tokens it would without deduplication. Figure 6 (right) shows that this compression translates directly to throughput gains when the GPU is compute-bound (batch size > 4): ReDrafter with tree attention significantly outperforms ReDrafter without it on both TPS and TPS×BSZ. At low batch sizes where compute is abundant, the overhead of processing extra tokens is hidden by parallelism, so the gain is negligible — which is exactly what you'd expect from removing a bottleneck: it matters only when the bottleneck is active.

This is a **modular algorithmic contribution** — the paper explicitly notes that dynamic tree attention "is not limited to ReDrafter" and can be used with any beam-search-based speculative decoding approach. It's not architecturally tied to the RNN draft model. This modularity strengthens the contribution: it's a reusable technique that other speculative decoding methods can adopt, independent of ReDrafter's other components.

---

### Innovation 4: Demonstrating That Speculative Decoding Architecture Choices Must Be Hardware-Aware, with Beam Width Serving as the Deployment-Time Knob That Bridges Server and On-Device Regimes

Most speculative decoding papers benchmark on a single hardware configuration (typically high-end NVIDIA GPUs) and report the speedup achieved with their fixed architecture. The implicit assumption is that a method that's fastest on an A100 will also be fastest on other hardware, or at least that the ranking of methods is hardware-invariant. ReDrafter's multi-platform evaluation — PyTorch on H100, MLX on M1 Max and M2 Ultra — challenges this assumption directly by showing that **the optimal configuration of the same method varies dramatically with hardware capability**.

The evidence is clear and striking. On an H100 GPU, optimal speedup comes at beam width 50+ (implied by the PyTorch experiments in Section 4.1, where Table 3 shows TPS continuing to increase up to beam width 64). On an M2 Ultra, optimal speedup comes at beam width 2–3 (Table 2). On an M1 Max, optimal speedup comes at beam width 1 (Table 2) — meaning beam search *at all* is counterproductive beyond the narrowest setting, because the GPU lacks the parallel compute to absorb the overhead. The same model, the same algorithm, three different optimal configurations spanning two orders of magnitude in beam width, driven entirely by GPU compute capacity.

What makes this an intellectual contribution rather than just an engineering observation is that it **identifies beam width as a first-class deployment parameter** that should be tuned per hardware target, not baked into the architecture. Prior speculative decoding methods (Medusa, EAGLE) have architectural parameters (number of heads, number of layers) that are fixed at training time — you can't change the number of Medusa heads at inference without retraining. ReDrafter's beam width is a pure inference-time parameter: you train the draft model once, and at deployment you set beam width based on profiling the target hardware. This decouples the training investment from the deployment optimization, which is practically important: one trained ReDrafter model can serve both server-grade and on-device deployments at near-optimal efficiency by simply adjusting the beam width dial.

This insight has implications beyond ReDrafter. It suggests that the speculative decoding literature's practice of comparing methods at their "best" settings on a single GPU may obscure important hardware-dependent effects. A method that looks competitive on an H100 might be impractical on a mobile GPU because its fixed architecture can't scale down to match limited compute. ReDrafter's tunable beam width makes it *inherently* more portable — and the paper's decision to benchmark on both server and mobile hardware makes this portability empirically visible.

This is a **conceptual contribution about evaluation methodology and deployment practice**, not a novel algorithm. It's incremental in the sense that the paper didn't invent the idea of tuning hyperparameters per hardware platform — but it's the first in the speculative decoding literature to demonstrate that this tuning is necessary, that the optimal configuration can vary by 50× across platforms, and that providing a tunable knob (rather than a fixed architecture) is a design desideratum for practical speculative decoding systems. The on-device results (up to 2.3× speedup on M2 Ultra with MLX) aren't just "we also ran on a phone" — they're evidence for the claim that hardware-aware design matters.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset and benchmarks.** The primary evaluation uses two dialogue benchmarks: **MT-Bench** (Zheng et al., 2024) and **AlpacaEval** (Dubois et al., 2024) — the latter referred to as "Alpaca" in the paper's tables and figures. MT-Bench is a multi-turn benchmark with 80 questions across 8 categories (writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities), where each question is a two-turn conversation. AlpacaEval is a single-turn benchmark with 805 questions evaluating instruction-following. The paper also reports per-category breakdowns on both benchmarks in Figure 5. The training data for the draft model is not explicitly specified, but is presumably derived from Vicuna's training data (the ShareGPT conversations used to fine-tune Vicuna), since the draft model is trained via knowledge distillation from the Vicuna LLMs. The exact size and composition of the training set are not reported — a notable omission for reproducibility.

- **Base models.** All experiments use the **Vicuna** family of models (7B, 13B, and 33B parameters). Vicuna models are LLaMA-based models fine-tuned on user-shared conversations from ShareGPT, making them representative of production dialogue LLMs. The paper states these models serve as the frozen LLMs whose inference ReDrafter accelerates. The choice of three model sizes enables analysis of how speculative decoding speedup scales with LLM size — an important dimension since memory-bandwidth bottlenecks become more severe as model size increases (more parameters to read from memory per forward pass), potentially increasing the relative benefit of reducing LLM calls. The paper does not report the exact Vicuna version (e.g., v1.3, v1.5), which is a minor ambiguity.

- **Metrics.** Two primary metrics are reported throughout:
  - **Speedup**: wall-clock speedup relative to autoregressive (AR) decoding, computed as `time(AR) / time(method)`. A speedup of 2.8× means the method generates tokens 2.8× faster than standard autoregressive generation.
  - **Tokens/Step**: the average number of tokens accepted by the LLM per generation step. This measures draft model accuracy independent of wall-clock overhead — higher Tokens/Step means the draft model is more accurately predicting what the LLM would generate. This metric is important because it separates the *accuracy* contribution (how many LLM calls are saved) from the *overhead* contribution (how much the draft model and verification cost). For the MLX on-device experiments, an additional metric **TPS (tokens per second)** is reported, measuring absolute generation throughput. For the batch-size sweep (Table 3), **TPS × BSZ** (per-request tokens-per-second multiplied by batch size) measures overall system throughput.

- **Baselines.** Three methods are compared in the PyTorch benchmark (Table 1):
  - **Autoregressive decoding**: the standard approach without speculative decoding. All speedup numbers are relative to this baseline.
  - **Medusa** (Cai et al., 2024): uses multiple independent draft heads attached to the LLM's hidden states, with a static tree attention pattern to handle the combinatorial explosion of candidate sequences. This is the primary architectural comparison since both Medusa and ReDrafter attach draft models to LLM hidden states rather than using separate models.
  - **EAGLE** (Li et al., 2024a): uses a recurrent draft model also conditioned on LLM hidden states, but with a different architecture and without beam search (EAGLE generates a single draft sequence autoregressively using its recurrent model, then optionally applies tree attention over multiple sampled sequences). EAGLE represents the "recurrent drafter" design space that ReDrafter competes in most directly.
  - The paper also implicitly compares against the original **Speculative Decoding** paradigm (Leviathan et al., 2023; Chen et al., 2023) in its related work discussion, but does not include these as baselines in the main experiments because they use detached draft models — a fundamentally different deployment model than the attached-draft-head approach shared by Medusa, EAGLE, and ReDrafter.

- **Generation budget / compute accounting.** The primary generation setting across all experiments is **temperature = 0** (greedy decoding). This is explicitly stated in Section 4.1: "We mainly conduct our experiments with temperature = 0, i.e., greedy decoding." For speculative decoding with greedy verification, the selection procedure is exact matching: the LLM verifier compares each draft token against its own greedy prediction and accepts the longest prefix where all tokens match. This is simpler and faster than rejection sampling (which requires computing and comparing probabilities at each position), and it eliminates stochasticity as a confounding variable when comparing methods. For the PyTorch experiments, the generation budget is implicitly controlled: each method is run under its own optimal configuration (beam width, number of heads, etc.) and compared on wall-clock time. For the MLX experiments (Table 2, Figure 7), beam width is explicitly varied (BW=1 through BW=4) while beam length is fixed at 4. For the batch-size sweep (Table 3), both beam width (1, 2, 4, 8, 16, 32, 64) and batch size (1 through 80) are systematically varied on MT-Bench with Vicuna 7B, measuring TPS and TPS×BSZ.

- **Hardware and implementation.** All PyTorch experiments run on a single **Nvidia H100 GPU**. The MLX experiments run on **Apple Silicon** chips: M1 Max and M2 Ultra, using Metal GPU acceleration. This dual-platform evaluation is unusual in the speculative decoding literature and is important for the paper's claim of hardware-adaptability. Implementation details for MLX are discussed in Appendix A.2, including lessons learned about lazy evaluation, JIT compilation, and dtype selection (float16 was found consistently faster than bfloat16 on Apple Silicon).

- **Cross-validation / statistical protocol.** The paper does not report any cross-validation or statistical significance testing. Speedup numbers are reported as point estimates without confidence intervals, standard deviations, or error bars. This is common in systems papers where wall-clock measurements are assumed to be low-variance given fixed hardware, but it means that small differences between methods (e.g., ReDrafter's 2.80× vs. EAGLE's 2.69× for Vicuna 7B on MT-Bench) should be interpreted cautiously — they may fall within measurement noise. The paper also does not specify how many runs were averaged to produce the reported speedups, or whether warm-up iterations were excluded. This is a minor methodological limitation.

---

### Main Quantitative Results

#### PyTorch Benchmark: Speedup and Tokens/Step on Server-Grade GPU

The headline result in Table 1: **ReDrafter achieves the highest speedup and Tokens/Step among all methods for Vicuna 7B and 13B across both MT-Bench and AlpacaEval, with a peak speedup of 2.80× on MT-Bench for both 7B and 13B models.** For Vicuna 33B, ReDrafter achieves the highest Tokens/Step (3.87) but slightly lower speedup than EAGLE (2.61× vs. 2.80×).

The specific numbers from Table 1 tell a nuanced story across model sizes and benchmarks:

**MT-Bench results (Table 1, top):**
- **Vicuna 7B**: ReDrafter 2.80× speedup, 4.20 Tokens/Step vs. EAGLE 2.69× / 3.96 vs. Medusa 2.39× / 2.55. ReDrafter leads EAGLE by 0.11× in speedup (a 4.1% relative improvement) and 0.24 in Tokens/Step (6.1% more tokens accepted per LLM call).
- **Vicuna 13B**: ReDrafter 2.80× / 4.21 vs. EAGLE 2.74× / 4.00 vs. Medusa 2.40× / 2.61. The pattern is nearly identical to 7B — ReDrafter and EAGLE are close, with ReDrafter having a marginal speedup advantage and a clear Tokens/Step advantage.
- **Vicuna 33B**: EAGLE 2.80× / 3.71 vs. ReDrafter 2.61× / 3.87 vs. Medusa 2.51× / 2.53. Here, EAGLE achieves higher speedup despite lower Tokens/Step, indicating that EAGLE's draft model overhead is lower for this model size — ReDrafter accepts more tokens per step but the beam search overhead costs more wall-clock time than EAGLE's approach, resulting in a net lower speedup.

**AlpacaEval results (Table 1, bottom):**
- **Vicuna 7B**: ReDrafter 2.69× / 4.06 vs. EAGLE 2.43× / 3.61 vs. Medusa 2.19× / 2.42. The gap between ReDrafter and EAGLE widens compared to MT-Bench — ReDrafter leads by 0.26× in speedup (10.7% relative improvement).
- **Vicuna 13B**: ReDrafter 2.78× / 4.02 vs. EAGLE 2.49× / 3.62. ReDrafter's advantage is again larger on AlpacaEval than MT-Bench.
- **Vicuna 33B**: EAGLE 2.59× / 3.29 vs. ReDrafter 2.43× / 3.61. The same pattern as MT-Bench for 33B: higher Tokens/Step for ReDrafter, lower speedup.

**What these numbers reveal about the methods:**

1. **ReDrafter consistently achieves higher Tokens/Step than EAGLE across all model sizes and benchmarks** (with the sole exception of parity at 33B MT-Bench where the numbers are very close: 3.87 vs. 3.71). This validates the central claim that beam search over the RNN draft model produces more accurate draft sequences (higher acceptance rates) — the Tokens/Step metric directly measures how many draft tokens the LLM accepts, and ReDrafter's multi-sequence beam search systematically finds longer accepted prefixes than EAGLE's approach.

2. **Higher Tokens/Step does not always translate to higher speedup.** At 33B, ReDrafter's beam search overhead (running the RNN across multiple beams, processing the wider packed beam in the LLM verification step) consumes enough additional wall-clock time to offset the benefit of accepting more tokens per step. This is a concrete manifestation of the compute-efficiency tradeoff the paper identifies: additional computation for wider beams must be parallelized effectively, and at larger model sizes, the LLM verification step becomes more expensive, eating into the gains from higher acceptance rates.

3. **Medusa trails both recurrent methods substantially**, particularly in Tokens/Step (2.42–2.61 vs. 3.61–4.21), confirming the paper's claim that independent predictions limit accuracy. Medusa's speedup of 2.19–2.51× is respectable but clearly below state-of-the-art, with the gap driven primarily by lower acceptance rates (fewer tokens accepted per LLM call means more LLM forward passes, reducing net speedup).

**Per-category analysis (Figure 5).** The paper reports speedup and Tokens/Step broken down by the 8 MT-Bench categories and AlpacaEval. ReDrafter "consistently performs well across all model sizes and dataset categories." The figure shows variation across categories — some categories (likely extraction, writing) may show higher Tokens/Step because they have more predictable, formulaic continuations, while others (reasoning, math, coding) may show lower Tokens/Step because they require more novel, less predictable generation. The paper does not provide a detailed per-category analysis beyond noting consistent performance, which is a missed opportunity to understand *which types of language* benefit most from the RNN+beam search approach.

**The gap between Tokens/Step and speedup.** The paper acknowledges this gap: "There's a gap between Tokens/Second and speedup, which is anticipated and arises from the overhead associated with the speculative decoding process." This overhead has multiple sources: the draft model's forward passes during beam search, the dynamic tree attention construction, and (crucially) the LLM verification forward pass over the packed beam — which processes more tokens than a standard autoregressive step even after deduplication, and more tokens as beam width increases. This overhead is what prevents Tokens/Step from translating 1:1 to speedup. The fact that ReDrafter achieves higher speedup despite this overhead demonstrates that the additional Tokens/Step from beam search outweighs the overhead cost — at least for 7B and 13B.

---

#### MLX-Based On-Device Inference: Speedup on Resource-Constrained Hardware

The headline result from Table 2: **ReDrafter achieves up to 2.28× speedup on M2 Ultra (Vicuna 33B, BW=4, beam length=4) and 1.32× speedup on M1 Max (Vicuna 7B, BW=1), with optimal beam width decreasing dramatically as GPU compute decreases — from BW=3 on M2 Ultra to BW=1 on M1 Max.**

Table 2 presents a systematic sweep of beam widths (1, 2, 3, 4) at fixed batch size 1 and beam length 4, with three Vicuna model sizes. The key patterns:

**M1 Max results:**
- **Vicuna 7B**: Peak speedup 1.32× at BW=1 (TPS=28.22, 2.15 Tokens/Step). Performance *declines* at higher beam widths: BW=2 gives 1.30× (TPS=27.69, 2.38 Tokens/Step), BW=3 gives 1.29× (TPS=27.54, 2.44 Tokens/Step), and BW=4 actually *slows down* relative to autoregressive at 0.94× (TPS=20.16, 2.44 Tokens/Step). Despite higher Tokens/Step at wider beams, the GPU cannot parallelize the additional computation fast enough, and total throughput drops.
- The paper notes: "We omitted experiments for the 13B and 33B models on the M1 Max, as they exceeded the device's memory capacity."

**M2 Ultra results:**
- **Vicuna 7B**: Peak speedup 1.52× at BW=3 (TPS=60.40, 2.44 Tokens/Step). Performance improves from BW=1 (1.43× / 2.15) to BW=2 (1.51× / 2.38) to BW=3 (1.52× / 2.44), then drops sharply at BW=4 (0.75× / 2.44). The Tokens/Step plateaus at BW=3 and 4 (2.44 in both cases) — meaning wider beams stop producing more accepted tokens — while overhead continues to increase, causing the BW=4 collapse.
- **Vicuna 13B**: Peak speedup 1.94× at both BW=2 and BW=3 (TPS≈43.52–43.55, 2.82 Tokens/Step). BW=1 achieves 1.87× (2.53 Tokens/Step) and BW=4 collapses to 0.97× (2.94 Tokens/Step). Again, Tokens/Step increases modestly from BW=2→4 (2.82→2.94) but the overhead penalty at BW=4 is catastrophic.
- **Vicuna 33B**: Peak speedup 2.28× at BW=4 (TPS=1.33, 2.56 Tokens/Step). This is the highest on-device speedup reported, but note the extremely low absolute TPS of 1.33 — the model is severely memory/IO-bound, and speculative decoding helps precisely because the LLM forward passes are so expensive relative to draft model computation. The speedup *increases monotonically* with beam width from BW=1 (1.97× / 2.17) to BW=4 (2.28× / 2.56) — unlike 7B and 13B, which peak at BW=3. This suggests the 33B model has *more* idle GPU compute available relative to the memory bottleneck, so the beam search overhead is more easily absorbed.

**The Tokens/Step vs. speedup tension explained.** Across all configurations, Tokens/Step increases monotonically with beam width — wider beams always find longer accepted prefixes. But TPS and speedup peak at an intermediate beam width and then crash. This is the central hardware-dependent tradeoff: **the LLM verification step for a wider beam processes more unique tokens** (even after dynamic tree attention removes shared prefixes, the packed beam grows with beam width). When the GPU has spare parallel compute (H100), this additional verification cost is absorbed. When the GPU is compute-saturated (M1 Max) or moderately loaded (M2 Ultra), the verification cost becomes a bottleneck, and beam width must be reduced to stay on the right side of the speedup equation.

**The dramatic TPS drop from 13B to 33B.** The paper observes: "We observed the TPS drops sharply from 43.55 to 1.33 when the model size increases from 13B to 33B, yet a significant speedup is still achieved." This is a critical observation for on-device deployment: larger models shift from compute-bound to memory/IO-bound, which actually *increases* the relative benefit of speculative decoding because each LLM forward pass becomes proportionally more expensive, but the draft model cost (which depends on the LLM's hidden state dimension, not the full parameter count) grows more slowly. The 2.28× speedup at 33B vs. 1.94× at 13B illustrates this — speculative decoding is more beneficial when the baseline is slower.

**The speedup landscape across beam width and beam length (Figure 7 in Appendix A.2).** The heatmaps in Figure 7 show TPS and speedup as a function of both beam width and beam length for Vicuna 7B on M1 Max and M2 Ultra. The key finding: **optimal beam length is close to the training length of 5** (optimal is at length 4–5), and **optimal beam width is hardware-dependent** (1 for M1 Max, 3 for M2 Ultra). The fact that speedup peaks at beam length 4 rather than 5 suggests the draft model's 5th-step predictions are less accurate, adding tokens that are rarely accepted while still incurring verification cost. This validates the paper's design choice of training with a 5-token prediction horizon — it provides headroom for the beam search to use 4–5 tokens effectively without wasting training compute on positions that are too inaccurate to be useful.

---

#### Beam Width and Batch Size Tradeoffs: Latency vs. Throughput

Table 3 presents a grid search over beam width (1, 2, 4, 8, 16, 32, 64) and batch size (1, 2, 4, 8, 10, 20, 40, 80) using Vicuna 7B on MT-Bench with the H100 GPU. This experiment investigates how ReDrafter scales when the GPU is asked to serve multiple requests simultaneously — a deployment scenario where overall system throughput matters as much as per-request latency.

**The latency-optimal configuration:** The highest per-request TPS (lowest latency) is achieved at **beam width 64, batch size 1** (TPS=110.64) or **beam width 64, batch size 2** (TPS=111.42). At batch size 1, TPS increases monotonically with beam width from 62.55 (BW=1) to 110.64 (BW=64) — nearly a 1.77× improvement in per-request throughput just from using wider beams. This demonstrates that on a powerful GPU with only one request to serve, the parallel compute is abundant enough to absorb beam widths up to 64, and the extra accepted tokens directly translate to lower latency.

**The throughput-optimal configuration:** The highest system throughput (TPS × BSZ) is achieved at **beam width 2, batch size 80** (1636.36) or **beam width 4, batch size 80** (1571.89). System throughput increases with batch size across all beam widths until hitting memory limits: at batch size 80, beam width 8 and above cause OOM. The best throughput configuration uses a surprisingly narrow beam (BW=2) — this is because at high batch sizes, the GPU is fully utilized processing the 80 concurrent requests, and the additional draft computation from wider beams competes for compute with the LLM verification work, reducing overall throughput even though individual requests accept more tokens per step.

**The interaction between beam width and batch size:**
- **At low batch sizes (1–4):** TPS increases with beam width. The GPU has spare parallel capacity, and wider beams convert this idle compute into faster per-request generation.
- **At moderate batch sizes (8–20):** The relationship becomes non-monotonic. For example, at batch size 8, TPS peaks at BW=16 (83.44) and declines at BW=32 (73.41) and BW=64 (54.19). The beam search overhead starts to compete with the LLM verification cost for GPU resources.
- **At high batch sizes (40–80):** TPS declines sharply with beam width. At batch size 80, BW=1 achieves 18.94 TPS but BW=4 only achieves 19.65 TPS (essentially flat), and BW=2 achieves 20.45 TPS — the narrowest beams are optimal because the GPU is already saturated with concurrent requests and any additional draft computation directly increases latency.

**Key deployment insight from Table 3:** The optimal configuration depends entirely on the deployment scenario. For **interactive, low-latency applications** (single-user chatbot, code assistant), use large beam width (32–64) and small batch size (1–2). For **high-throughput batch processing** (evaluating benchmarks, generating training data), use small beam width (2–4) and large batch size (40–80). The paper concisely states: "For scenarios prioritizing low latency, a larger beam width with a smaller batch size is recommended. Conversely, if high throughput is the main goal, a larger batch size paired with a moderate beam width is more effective."

**Memory constraints.** At batch size 80, out-of-memory (OOM) errors occur for beam widths 8 and above. This is because wider beams produce larger packed beams (more unique tokens after deduplication) that must be stored in the LLM's key-value cache during verification. The KV cache memory scales with `batch_size × sequence_length × num_layers × hidden_dim`, and the packed beam's effective sequence length grows with beam width. This is a practical constraint: on memory-limited hardware, the beam width cannot be arbitrarily increased even if compute is available.

---

#### Dynamic Tree Attention: Compression and Throughput Gains

The experiments in Figure 6 quantify the benefit of dynamic tree attention in two ways: (1) the token compression ratio achieved when deduplicating beam search results, and (2) the throughput impact when the GPU is under compute pressure.

**Compression ratio (Figure 6, left):** The experiment sweeps beam width from 5 to 70 (producing 25–350 total draft tokens) on MT-Bench with Vicuna 7B, batch size 1, beam length 5. The median compression reduces the number of tokens by **30–60%**, with the 1st percentile showing ~20% reduction and the 99th percentile showing ~50–65% reduction. The compression ratio is "consistent across different beam sizes" — meaning that as beam width increases, the number of unique tokens in the packed beam grows sub-linearly. For example, doubling beam width does not double the packed beam size because wider beams discover more shared prefixes among candidates. This sub-linear scaling is what makes wide beams practical: without it, the LLM verification cost would scale linearly with beam width and quickly eat into speedup gains.

**Throughput impact under compute pressure (Figure 6, right):** The experiment fixes beam width at 45, beam length at 5, and varies batch size from 1 to 100. Two configurations are compared: ReDrafter with dynamic tree attention ("RD w TA") and ReDrafter without ("RD w/o TA"). At batch sizes 1–4, the TPS and TPS×BSZ are nearly identical between the two — the GPU has abundant compute to process the duplicated tokens without slowing down. At batch sizes above 4, a compute bottleneck emerges: "ReDrafter with dynamic tree attention (RD w TA) significantly outperforms ReDrafter without tree attention, delivering higher throughput and more tokens per second." The gap widens as batch size increases because the GPU is increasingly saturated, and the 30–60% reduction in tokens to verify translates directly to higher throughput.

**Why this matters for practical deployment:** In production LLM serving, batch sizes are often dynamically adjusted based on request load. During periods of high load, the system operates at the throughput-optimal batch size where GPU compute is fully utilized. In this regime, dynamic tree attention provides its maximum benefit — it reduces the verification cost per request, allowing the system to process more requests per second or serve the same number of requests with lower latency. During periods of low load (batch size 1–2), the benefit is minimal, but the overhead of dynamic tree attention (the `dedup_prefix` and `pack_beam` operations) is also minimal (the paper notes these are efficient tensor operations). So the feature is effectively "free" during low-load periods and highly beneficial during high-load periods — ideal for production serving.

---

#### Knowledge Distillation: Ablation on Draft Model Training Objective

Table 4 compares two draft models for Vicuna 7B: one trained with knowledge distillation (LLM-generated targets) and one trained with ground-truth token targets (standard next-token prediction). The sweep covers beam widths 1, 2, 4, 16, and 64, measuring speedup and Tokens/Step on MT-Bench with batch size 1.

**The headline finding:** "Distillation lead to an approximate 10% increase in the speedup and the average accepted tokens per step." The specific numbers:

| Beam Width | No Distill Speedup | Distill Speedup | Relative Gain | No Distill T/S | Distill T/S |
|---|---|---|---|---|---|
| 1 | 1.47× | 1.54× | +4.8% | 2.21 | 2.35 |
| 2 | 1.52× | 1.60× | +5.3% | 2.31 | 2.50 |
| 4 | 1.54× | 1.72× | +11.7% | 2.48 | 2.73 |
| 16 | 1.80× | 1.92× | +6.7% | 2.87 | 3.09 |
| 64 | 1.99× | 2.18× | +9.5% | 3.30 | 3.58 |

The gain is remarkably consistent: **at every beam width, distillation improves both speedup and Tokens/Step**, with speedup gains ranging from 4.8% to 11.7% and Tokens/Step gains ranging from 0.14 to 0.28 tokens per step. The absolute improvement in Tokens/Step grows slightly with beam width (0.14 at BW=1 vs. 0.28 at BW=64), suggesting that distillation is particularly helpful when the beam search is exploring more diverse candidates — a distillation-trained draft model spreads probability mass more similarly to the LLM, meaning the beam contains candidates the LLM is more likely to accept.

**Why this matters beyond the 10% number:** The distillation result validates that draft model training objective matters substantially — a 10% speedup improvement from changing only the training targets (not the model architecture, not the beam search, not the hardware) is significant. It also demonstrates that "aligning with the LLM" is not just theoretical — it has measurable, consistent impact on the core metrics. The paper's framing of this as "investing more resource in training time" to improve inference efficiency is empirically justified: the one-time cost of generating distillation data (one LLM forward pass per training position) pays off with persistent inference-time speedup.

**Limitation of this ablation:** The experiment only compares two extremes — pure ground-truth training vs. pure distillation — and does not explore intermediate regimes (e.g., mixing ground-truth and distillation losses, using different temperatures for the LLM when generating distillation targets, or fine-tuning a ground-truth-trained model with distillation). The choice of temperature 0 for generating distillation targets is also not ablated — using a positive temperature might produce softer targets that better capture the LLM's full distribution, potentially improving the draft model's ability to match the LLM at non-greedy temperatures, but this is not tested.

---

### Ablation Studies and Robustness Checks

**Beam width sweep across batch sizes (Table 3):** The full grid search over `beam_width ∈ {1, 2, 4, 8, 16, 32, 64}` and `batch_size ∈ {1, 2, 4, 8, 10, 20, 40, 80}` demonstrates two regimes of operation. At low batch sizes, per-request TPS scales monotonically with beam width up to the maximum tested (110.64 TPS at BW=64, BSZ=1 vs. 62.55 TPS at BW=1). At high batch sizes, this relationship reverses: at BSZ=80, BW=2 achieves the highest TPS (20.45) with wider beams performing worse or OOM. This confirms that beam width's optimal value is not just hardware-dependent but also **batch-size-dependent** — a finding that complicates deployment but provides flexibility.

**Beam width sweep on Apple Silicon (Table 2, Figure 7):** The optimal beam width varies from 1 (M1 Max, 7B) to 3 (M2 Ultra, 7B and 13B) to 4 (M2 Ultra, 33B). At all configurations, beam width 4 causes a collapse in speedup, with the 7B M2 Ultra dropping from 1.52× at BW=3 to 0.75× at BW=4 — a speedup below 1.0× means the speculative decoding is *slower* than autoregressive. This sharp phase transition at the hardware's compute limit is a non-obvious finding: the penalty for exceeding the optimal beam width is not gradual but catastrophic, because the verification step's cost grows with the packed beam size and can overwhelm any benefit from additional accepted tokens.

**Beam length sweep (Figure 7):** On both M1 Max and M2 Ultra with Vicuna 7B, optimal beam length is 4, not 5 — one step shorter than the training horizon. The paper explains this: "the RNN may not always accurately predict the 5th token." This is an implicit finding about the draft model's accuracy degradation at longer horizons, and it suggests that training with a longer horizon (e.g., 8 tokens) might not help — the accuracy drop-off might simply move to later positions. The beam length is constrained by both training horizon and inherent prediction difficulty at long ranges.

**Dynamic tree attention vs. no tree attention (Figure 6, right):** At low batch sizes, the two configurations are essentially identical. At high batch sizes, tree attention provides substantial gains. This is a robustness check in the sense that it shows dynamic tree attention's overhead is negligible when not needed and its benefit is large when needed — exactly the property you want from an optimization that addresses a bottleneck.

**Distillation vs. no distillation across beam widths (Table 4):** The gain is consistent across all beam widths (4.8–11.7% speedup improvement), with no beam width where distillation hurts. This robustness check validates that distillation is uniformly beneficial, not just beneficial in some regime. It also shows the Tokens/Step improvement is consistent (0.14–0.28 absolute gain), confirming that distillation improves the fundamental alignment of the draft model with the LLM, not just some secondary effect.

**Model size scaling (Table 1, Table 2):** ReDrafter's performance is tested across three model sizes in both PyTorch (7B, 13B, 33B) and MLX (7B, 13B, 33B on M2 Ultra, 7B only on M1 Max). In PyTorch, speedup is nearly identical for 7B (2.80×) and 13B (2.80×) but drops to 2.61× for 33B. In MLX, speedup *increases* with model size: 1.52× (7B), 1.94× (13B), 2.28× (33B) for the optimal beam width on M2 Ultra. These opposite trends are explained by the different bottlenecks: on the H100, the 33B model's larger verification cost reduces net speedup; on the M2 Ultra, the 33B model's severe memory bottleneck makes autoregressive decoding so slow (TPS=1.33 at best) that even a modest reduction in LLM calls produces a large relative speedup. This cross-platform inversion of the model-size scaling trend is a non-obvious finding that underscores the hardware-dependence of speculative decoding efficiency.

**Benchmark diversity (Figure 5):** ReDrafter is evaluated across 8 MT-Bench categories plus AlpacaEval, and "consistently performs well across all model sizes and dataset categories." While the paper does not provide per-category numbers, the consistent trend across diverse task types (writing, coding, math, reasoning, roleplay, etc.) suggests the method is not overfit to a particular text genre.

**MLX-specific implementation lessons (Appendix A.2.2):** The paper documents three non-obvious implementation findings: (1) float16 is consistently faster than bfloat16 on Apple Silicon (unlike on CUDA where bfloat16 often has advantages); (2) MLX's lazy evaluation means model parameters are loaded lazily — you must call `mlx.core.eval(model.parameters())` before benchmarking; (3) frequent calls to `array.item()` break JIT compilation and significantly degrade performance. These are practical robustness findings for practitioners implementing speculative decoding on Apple Silicon.

---

### Critical Assessment

#### Claim 1: ReDrafter achieves state-of-the-art speedup for LLM inference

The paper claims ReDrafter "achieves state-of-the-art speedup" (Abstract) and that in the PyTorch benchmark, "ReDrafter attains the highest speedup and Tokens/Step with Vicuna 7B and 13B" (Section 4.1).

**What the experiments demonstrate:** ReDrafter achieves 2.80× speedup on MT-Bench for Vicuna 7B and 13B, compared to EAGLE's 2.69× (7B) / 2.74× (13B) and Medusa's 2.39× (7B) / 2.40× (13B). The margin over EAGLE is 0.11× for 7B and 0.06× for 13B — differences of 4.1% and 2.2% respectively. On AlpacaEval, the gap is larger: 0.26× (10.7%) for 7B and 0.29× (11.6%) for 13B.

**Are these margins meaningful?** The paper does not report measurement variance, error bars, or statistical tests. On fixed hardware, wall-clock measurements for LLM inference are generally low-variance (the computation is deterministic with temperature 0), but there can be variation from GPU clock speed fluctuations, memory access patterns, and system-level noise. A 2.2% improvement (Vicuna 13B on MT-Bench) could plausibly fall within measurement noise. The larger margins on AlpacaEval (10.7% and 11.6%) are more convincing, but without error bars, it's impossible to say with certainty.

**What's genuinely state-of-the-art:** The Tokens/Step numbers are clearly and consistently higher for ReDrafter than all baselines. On MT-Bench: 4.20 (7B), 4.21 (13B), 3.87 (33B) vs. EAGLE's 3.96, 4.00, 3.71. The Tokens/Step advantage is 0.16–0.24 across all model sizes — too large and too consistent to be measurement noise, and it directly validates the paper's core architectural claim that beam search over an RNN draft model captures more accepted tokens per step than EAGLE's approach.

**The 33B exception is important and under-discussed.** EAGLE achieves 2.80× speedup on Vicuna 33B vs. ReDrafter's 2.61×, despite ReDrafter having higher Tokens/Step (3.87 vs. 3.71). This means ReDrafter's overhead is larger at 33B, and the paper's "state-of-the-art" claim must be qualified: it holds for 7B and 13B but *not* for 33B. This is partially hidden by the aggregate framing — a reader skimming the abstract might miss this exception.

**Missing baselines:** The paper does not compare against several speculative decoding methods that were published before this paper's arXiv submission (March 2024): SpecInfer (Miao et al., 2023), which also uses tree-based verification; Staged Speculative Decoding (Spector & Re, 2023); or the original speculative sampling with a separate draft model (Leviathan et al., 2023; Chen et al., 2023). While these methods use different deployment paradigms (detached draft models), a comparison would contextualize ReDrafter's speedup relative to the broader speculative decoding landscape, not just the attached-draft-head subfield.

**Assessment:** The claim of state-of-the-art speedup is supported with qualifications. ReDrafter clearly leads on 7B and 13B models on both MT-Bench and AlpacaEval, with Tokens/Step advantages that validate its architectural approach. The 33B result where EAGLE leads tempers the universality of the claim, and the small speedup margin on 13B MT-Bench (2.2%) may not be significant without error bars. The claim would be stronger with statistical characterization of measurement variance and with comparisons to detached draft model baselines.

---

#### Claim 2: ReDrafter's RNN draft model with beam search provides higher prediction accuracy (more accepted tokens) than prior approaches

This is the central architectural claim. The paper argues that Medusa's independent heads fail to leverage sequential structure, and that EAGLE's recurrence is beneficial but doesn't exploit beam search for diversity.

**What the experiments demonstrate:** The Tokens/Step metric directly measures prediction accuracy in the speculative decoding context. ReDrafter achieves 4.20 Tokens/Step (7B MT-Bench) vs. EAGLE's 3.96 and Medusa's 2.55. This is a 6.1% improvement over EAGLE and a 64.7% improvement over Medusa. The gap vs. Medusa is massive and clearly attributable to sequential vs. independent prediction; the gap vs. EAGLE is smaller but consistent, attributable to beam search exploration.

**What's missing:** The paper does not ablate the beam search away to isolate the RNN architecture's contribution. Specifically, an experiment comparing ReDrafter with beam width 1 (single greedy draft sequence, no beam search) against EAGLE would reveal how much of ReDrafter's Tokens/Step advantage comes from the RNN architecture alone vs. from beam search. Table 2 shows ReDrafter BW=1 achieves 2.15–2.53 Tokens/Step in MLX experiments, but these numbers are not directly comparable to the PyTorch EAGLE numbers in Table 1 due to different hardware and potentially different draft model training. A controlled comparison of RNN-without-beam-search vs. EAGLE on the same hardware with the same training data would strengthen this claim.

**Assessment:** The claim is strongly supported for ReDrafter vs. Medusa (the sequential vs. independent distinction is clear and the performance gap is large). For ReDrafter vs. EAGLE, the claim is supported directionally (consistent Tokens/Step advantage) but the mechanism attribution is incomplete — we cannot separate how much of the advantage comes from the RNN design vs. from beam search vs. from training differences. The paper would benefit from an ablation that isolates beam search as the differentiating factor.

---

#### Claim 3: Dynamic tree attention is an efficient, GPU-friendly algorithm that reduces verification overhead

**What the experiments demonstrate:** Figure 6 (left) shows 30–60% token compression at median, with the compression ratio consistent across beam widths. Figure 6 (right) shows that this compression translates to throughput gains when the GPU is compute-bound (batch size > 4). The algorithm itself is presented in Appendix A.1 with a concise tensor-based implementation.

**What's missing:** The paper does not compare dynamic tree attention against alternative deduplication approaches — for example, a sequential trie-based implementation, or the static tree attention used in Medusa and SpecInfer. Without this comparison, we cannot quantify how much the GPU-friendliness matters vs. simply doing any form of deduplication. The paper also doesn't report the wall-clock overhead of the `dedup_prefix` and `pack_beam` operations themselves — while described as "GPU-friendly" and "with little overhead," no timing measurements are provided. For a systems paper, overhead quantification is essential.

**Assessment:** The compression ratio data (Figure 6, left) convincingly shows that deduplication removes substantial redundancy from beam search outputs. The throughput data (Figure 6, right) shows that this matters in compute-constrained regimes. The algorithmic contribution — expressing prefix deduplication as parallel tensor operations — is clever and well-documented in the appendix. However, the claim that it's "efficient" is qualitative without overhead measurements or comparisons to alternative implementations. This is a moderate weakness in an otherwise well-executed systems evaluation.

---

#### Claim 4: ReDrafter is practical for on-device deployment, achieving speedup on resource-constrained Apple Silicon GPUs

**What the experiments demonstrate:** ReDrafter achieves 1.32× speedup on M1 Max (7B), 1.52× on M2 Ultra (7B), 1.94× on M2 Ultra (13B), and 2.28× on M2 Ultra (33B) — all with an MLX implementation. The paper argues this validates "its capability to optimize performance in resource-constrained environments" and "significant potential for further improvement as on-device hardware continues to evolve."

**Are these speedups meaningful on-device?** A 1.32× speedup on M1 Max means that inference takes about 75% of the original time. For a model that takes 10 seconds to generate a response, this saves about 2.5 seconds — noticeable but not transformative. The 2.28× speedup on M2 Ultra for 33B is more impactful, but the absolute TPS of 1.33 means the model generates only about 80 tokens per minute — far too slow for interactive use. The paper acknowledges this: "for larger models, compression techniques like quantization may be necessary to achieve acceptable latency."

**What's notably missing from the on-device evaluation:**
- **No baseline comparison on Apple Silicon.** The paper does not compare against Medusa or EAGLE on M1 Max or M2 Ultra. The on-device results are reported as standalone speedups over autoregressive decoding, not as comparisons to alternative speculative decoding methods. We cannot assess whether ReDrafter is the *best* on-device speculative decoding method or simply *a* working one.
- **No power or thermal measurements.** On-device deployment is constrained not just by compute but by battery life and thermal throttling. The paper does not report power consumption, GPU temperature, or whether the observed speedups are sustainable (or if thermal throttling would reduce them during extended use).
- **No latency distribution.** Mean TPS is reported, but interactive applications care about tail latency — the occasional slow generation is more noticeable than the average speed. Percentile distributions (p50, p95, p99) of per-token or per-request latency are not provided.

**Assessment:** The on-device results demonstrate feasibility — ReDrafter *works* on Apple Silicon and provides non-trivial speedup. However, the claim of "practicality" is only partially supported. For 33B models, the absolute throughput is too low for interactive use regardless of speedup. For 7B and 13B on M2 Ultra, speedups of 1.5–1.9× are meaningful but not characterized in terms of user experience (latency percentiles, response time distributions). The lack of baseline comparisons on Apple Silicon is a significant gap — we don't know whether Medusa or EAGLE would achieve similar or better on-device speedups if implemented in MLX.

---

#### Claim 5: Knowledge distillation improves speculative decoding performance

**What the experiments demonstrate:** Table 4 shows a consistent ~10% improvement in both speedup and Tokens/Step across all beam widths (1–64) when the draft model is trained with LLM-generated targets vs. ground-truth targets. The improvement is monotonic and present at all tested beam widths.

**What's missing:**
- **No distillation temperature ablation.** The paper uses temperature 0 for generating distillation targets, producing hard (one-hot) LLM predictions. Standard knowledge distillation often benefits from higher temperatures (softening the teacher's distribution to provide richer training signal). The paper doesn't explore whether temperature > 0 improves results.
- **No data scale ablation.** We don't know how much distillation data was used, or whether the 10% gain would change with more or less data. If the gain requires generating distillation targets for every position of a massive training corpus, the training cost might be substantial.
- **No comparison to DistillSpec or other distillation approaches.** While the paper cites Zhou et al. (2023) (DistillSpec) and notes that Medusa2 also uses distillation, it doesn't compare ReDrafter's distillation approach against these prior methods.

**Assessment:** The claim is well-supported directionally — distillation helps, and the effect is consistent. The 10% improvement is practically meaningful. However, the experiment is a binary ablation (with vs. without) rather than a thorough investigation of distillation methodology. The optimal distillation strategy (temperature, data quantity, loss function) remains unexplored.

---

#### Missing experiments that would strengthen the paper

Several experiments are conspicuously absent and would significantly strengthen the paper's claims:

1. **Stochastic decoding (temperature > 0) evaluation.** All reported experiments use greedy decoding (temperature = 0). This is justified as a controlled setting, but most production LLM deployments use positive temperatures (typically 0.6–1.0) for diversity. ReDrafter's performance with rejection sampling (mentioned as an alternative to greedy matching in Section 3.4) is never evaluated. Since rejection sampling changes the acceptance dynamics (tokens can be accepted probabilistically even when they don't match the LLM's top prediction), the speedup and Tokens/Step may differ substantially from the greedy case.

2. **Draft model size and FLOP characterization.** The paper never specifies the draft model's parameter count, architecture dimensions, or inference FLOPs relative to the LLM. This makes it impossible to assess whether ReDrafter's speedup comes from an efficient draft model or from a draft model that is simply larger (and thus more accurate but also more expensive) than EAGLE's or Medusa's. A fair comparison would report draft model size and inference cost alongside speedup.

3. **End-to-end latency measurement.** Speedup is reported as `time(AR) / time(method)`, but it's unclear whether this includes all overhead: prompt processing (prefill), draft model initialization, dynamic tree attention construction, and the non-speculative generation steps at the beginning of each sequence (when there's no LLM hidden state to condition on yet). If speculative decoding adds significant latency to the first few tokens (prefill phase), the user-perceived speedup for short responses might be much lower than the average over full sequences.

4. **Comparison against detached draft models.** A smaller Vicuna variant (or a separately trained small LLaMA) used as a detached draft model would provide a baseline for whether ReDrafter's attached-draft-head approach is genuinely necessary or if similar speedups can be achieved with "simpler" separate-model speculative decoding. The paper's argument that separate models add "complexity to their integration within a unified serving system" (Section 2) is a deployment argument, not a performance argument — it doesn't replace a head-to-head speedup comparison.

5. **Training cost quantification.** The paper frames distillation as "investing more resource in training time," but never quantifies the training cost. How many GPU-hours were needed to generate the distillation data? How long did draft model training take? How do these costs compare to the inference-time savings over the model's expected deployment lifetime? Without these numbers, the "investment" framing is qualitative.

6. **Statistical characterization of measurements.** All speedup and Tokens/Step numbers are point estimates. Reporting standard deviations across multiple runs, or percentile distributions of per-step Tokens/Step, would allow readers to assess whether small differences between methods are significant.

#### Overall Assessment

The experimental evaluation is well-designed for its primary purpose: demonstrating that ReDrafter achieves competitive or leading speedup on standard benchmarks compared to the most relevant prior methods (Medusa, EAGLE). The dual-platform evaluation (H100 + Apple Silicon) is a genuine strength that provides evidence for hardware-adaptability — a claim most speculative decoding papers don't make or evaluate. The beam width sweep experiments (Tables 2 and 3, Figure 7) are thorough and reveal non-obvious interactions between beam width, batch size, and hardware capability.

However, the evaluation has clear limitations: (1) evaluation is limited to greedy decoding, while production deployments use stochastic sampling; (2) no statistical characterization of measurement variance; (3) missing comparisons to detached draft model baselines and to prior methods on Apple Silicon; (4) no quantification of draft model size/FLOPs or training cost; (5) the "state-of-the-art" claim must be qualified for 33B models. These limitations don't invalidate the paper's contributions, but they mean the claims should be interpreted as applying specifically to greedy-decoding speculative decoding with attached draft heads on the tested hardware, not as universal statements about speculative decoding optimality.

## 6. Limitations and Trade-offs

### The Greedy-Decoding Evaluation Gap: All Reported Results Assume Temperature = 0, Leaving Production Behavior Uncharacterized

**The assumption or constraint.** The paper explicitly restricts its main evaluation to greedy decoding: "We mainly conduct our experiments with temperature= 0, i.e., greedy decoding" (Section 4.1). The selection procedure in this regime is exact matching — a draft token is accepted if and only if it matches the LLM's top-1 prediction. The paper mentions that "the selection method can range from a greedy approach (aka. token matches), to rejection sampling" (Section 3.4), but **nowhere evaluates performance with stochastic decoding**, rejection sampling, or any temperature > 0. All speedup numbers in Tables 1–4, Figures 5–7, and the MT-Bench/AlpacaEval results reflect only the greedy regime.

**The consequence.** This is a first-order limitation because essentially all production LLM deployments use positive temperatures (typically 0.6–1.0) to produce diverse, natural-sounding outputs. With rejection sampling, the acceptance dynamics change qualitatively: a draft token can be accepted probabilistically even when it does not match the LLM's greedy prediction, as long as `p_LLM(x) / p_draft(x)` exceeds a random threshold. This means Tokens/Step could be higher (more tokens accepted per step due to probabilistic acceptance) or lower (if the draft model's distribution diverges from the LLM's in ways that rejection sampling penalizes). The beam search strategy — which selects candidates to maximize draft model probability — may also need adjustment for stochastic decoding, where matching the LLM's *distribution* matters more than matching its *mode*. A practitioner deploying ReDrafter in a chatbot with temperature 0.7 has **no empirical guidance** on what speedup to expect, whether the optimal beam width changes, or whether the relative ranking against EAGLE or Medusa holds. The omission is particularly significant given that the reference paper on speculative decoding (Leviathan et al., 2023) was explicitly designed for and evaluated with stochastic sampling — ReDrafter's evaluation on temperature 0 alone is a deviation from the standard speculative decoding evaluation protocol.

**What evidence exists in the paper.** None. The paper provides zero experimental results with temperature > 0, rejection sampling, or any stochastic decoding variant. This is a pure omission, not an under-explored result. The training data for knowledge distillation was also generated at temperature 0 ("the LLM generates 5 future tokens at each position of the ground-truth response using a temperature of 0," Section 4.3.3), meaning the draft model was never trained to match the LLM's distribution at positive temperatures — it was trained to match the LLM's greedy mode. Whether this draft model can effectively support rejection sampling (which requires the draft model's full distribution to be reasonably calibrated to the LLM's, not just to share the same argmax) is entirely unknown.

**Mitigation status.** The paper does not acknowledge this as a limitation. The phrase "temperature= 0, i.e., greedy decoding" is presented as a neutral experimental choice, not as a scope restriction. There is no discussion of how results might change with stochastic decoding, no suggestion that future work should address this, and no caveat in the abstract or conclusions about the evaluation regime. The mention of rejection sampling in Section 3.4 is purely descriptive ("the selection method can range from...") without any empirical follow-through. This is the most consequential gap in the paper for practitioners, because the leap from "2.8× speedup at temperature 0" to "2.8× speedup in my production chatbot" is unsupported.

---

### The Difficulty Estimation Overhead Is Transferred to Training-Time Data Generation, But the Paper Never Quantifies the Cost It Claims to Be "Investing"

**The assumption or constraint.** The paper frames knowledge distillation as an economic tradeoff: "ReDrafter applies knowledge distillation from LLMs, improving inference time efficiency by investing more resource in training time" (Section 2). The implication is that a one-time training cost is amortized over future inference savings. However, the paper **never reports** any training cost: not the amount of distillation data generated, not the GPU-hours required for data generation (running the full LLM forward pass at every position of every training sequence), not the draft model training time, and not the total training compute relative to inference savings. The training data generation procedure (Section 3.5) requires the LLM to generate `T` tokens at every position of every training sequence — for a corpus with millions of tokens, this means millions of LLM forward passes just to create the training data, each of which is exactly the expensive operation speculative decoding aims to avoid at inference time.

**The consequence.** The "investment" framing is qualitative and unverifiable. A deployment team considering ReDrafter cannot assess whether the training cost is acceptable — it might be trivially small (a few GPU-hours on a single machine) or impractically large (requiring a cluster comparable to the LLM's own pretraining or fine-tuning). Without cost quantification, the claimed benefit of distillation (the ~10% speedup improvement in Table 4) cannot be weighed against its cost. Worse, the distillation cost scales with the size of the training corpus and the LLM's forward-pass cost — for a very large LLM (e.g., 70B parameters) or a very large training dataset, generating distillation targets could cost more than the inference-time savings over the model's entire deployment lifetime. The paper provides no breakeven analysis to guide practitioners on when distillation is worth the investment.

A secondary consequence: the difficulty estimation problem from the reference paper (Section 3.2 — generating 2048 samples per question to estimate difficulty) has an analogue here. The distillation data generation requires running the LLM on every training position — how was this cost managed? Was the training data subsampled? Was distillation applied to all of Vicuna's training data or only a subset? The paper's silence on these questions means the reported results may depend on an unreported, potentially expensive data generation pipeline that is not described sufficiently for reproduction.

**What evidence exists in the paper.** None. The only training cost mention is the qualitative statement about "investing more resource in training time" (Section 2). Appendix A.1 discusses implem`entation, not training cost. The paper provides no wall-clock time, GPU-hour count, or FLOP estimate for either distillation data generation or draft model training. The model architecture hyperparameters (draft model size, number of MLP layers, hidden dimensions, training epochs, learning rate, batch size) are also largely unspecified — Section 3.5 mentions "a few layers of MLPs" and the Adam optimizer but gives no concrete numbers. This is a gap in both cost accounting and reproducibility.

**Mitigation status.** The paper does not acknowledge this as a limitation. Future work (Section 5) mentions "enhancing draft model training through more advanced distillation techniques" and "optimizing implementation to ensure consistent performance gains and less overhead," but these refer to improving the method further, not to quantifying the cost of the current approach. The absence of training cost quantification is particularly notable because the paper's framing of "investing at training time to save at inference time" is central to its narrative — without numbers, the framing is a slogan, not an empirically supported claim.

---

### The 33B Model Size Anomaly: ReDrafter Loses to EAGLE on the Largest Tested Model, Challenging the Generality of the Architectural Advantage

**The assumption or constraint.** The paper claims ReDrafter "achieves state-of-the-art speedup" (Abstract) and "delivers state-of-the-art speedups across various implementations and hardware platforms" (Section 2). This claim is supported for Vicuna 7B and 13B, but fails for Vicuna 33B on the PyTorch benchmark — the largest model tested and the one where speculative decoding's benefits should be most pronounced (since larger models are more severely memory-bandwidth-bound, making each saved LLM call more valuable). Table 1 shows that EAGLE achieves 2.80× speedup on Vicuna 33B (MT-Bench) vs. ReDrafter's 2.61×, despite ReDrafter having higher Tokens/Step (3.87 vs. 3.71). On AlpacaEval, the gap is 2.59× (EAGLE) vs. 2.43× (ReDrafter). **ReDrafter is not state-of-the-art for 33B models.**

**The consequence.** The 33B result reveals a scaling limitation: ReDrafter's beam search overhead grows with model size in a way that EAGLE's approach does not. ReDrafter accepts *more* tokens per LLM call even at 33B (3.87 vs. 3.71), but the wall-clock cost of verifying those tokens — the LLM forward pass over the packed beam — is large enough that EAGLE's lower-overhead approach yields higher net speedup. This is a fundamental architectural tradeoff: beam search improves draft accuracy (higher Tokens/Step) but increases verification cost (the packed beam contains more unique tokens than EAGLE's draft sequences). At 7B and 13B, the accuracy gain outweighs the verification cost; at 33B, the balance tips. Whether this tradeoff continues to worsen at 70B, 130B, or larger models is entirely unknown from the paper's data — and since the trend is toward deploying ever-larger models, the 33B result may be the most practically relevant data point for understanding ReDrafter's scaling behavior.

This limitation directly challenges the paper's implicit claim that beam search is unambiguously beneficial. The 33B result suggests that beam search's value is model-size-dependent, and there may exist a crossover point beyond which narrower beams (or no beam search at all, as in EAGLE) are optimal even on powerful GPUs. The paper does not investigate this crossover or provide guidance on how to predict it.

**What evidence exists in the paper.** Table 1 (bottom rows for 33B) directly shows the crossover. The Tokens/Step advantage for ReDrafter is real (3.87 vs. 3.71 on MT-Bench, 3.61 vs. 3.29 on AlpacaEval), but the speedup numbers invert. The paper acknowledges this only implicitly by noting ReDrafter's speedup is "slightly lower than EAGLE's" for 33B (Section 4.1), without analyzing why or what it implies. Table 3 and the on-device results in Table 2 provide additional context: on M2 Ultra, the beam width trend actually reverses (wider beams continue to help at 33B, with speedup increasing from 1.97× at BW=1 to 2.28× at BW=4), suggesting the crossover depends on both model size and hardware capability in complex ways. The paper does not synthesize these observations into a coherent picture of ReDrafter's scaling behavior.

**Mitigation status.** The paper does not treat the 33B result as a limitation. It is reported as a data point without analysis. The "state-of-the-art" framing in the abstract and introduction does not qualify that this claim holds for 7B and 13B but not 33B. Future work (Section 5) mentions "optimizing implementation to ensure consistent performance gains and less overhead," which is vague. A thorough investigation of how ReDrafter's speedup scales with model size — and whether the beam width can be reduced at larger model sizes to recover the speedup advantage — is absent.

---

### Draft Model Architecture and Training Details Are Insufficient for Reproduction or Fair Comparison with Prior Methods

**The assumption or constraint.** The paper describes the draft model as "a few layers of MLPs with skip connections, followed by a standard softmax layer" (Section 3.1) using "a simple recurrent design" with "one layer RNN to make the model simple" (Section 3.1). However, it does not specify: the number of MLP layers, the hidden dimension of the MLP, the hidden dimension of the RNN state `s_t`, the activation function `f`, the total parameter count of the draft model, the training hyperparameters (learning rate, batch size, number of epochs, optimizer settings), the size or source of the training dataset, or the inference FLOPs of the draft model relative to the LLM. These are not minor omissions — they prevent independent reproduction and make it impossible to assess whether ReDrafter's speedup advantage over EAGLE and Medusa comes from a better architecture or simply from a larger/more expensive draft model.

**The consequence.** Without draft model size and FLOP characterization, the paper's central comparison (ReDrafter vs. EAGLE vs. Medusa) is fundamentally incomplete. All three methods add computational overhead to the LLM — the draft model forward passes, the beam search, the verification step over extra tokens. If ReDrafter's draft model has 2× more parameters than EAGLE's, then its higher Tokens/Step could be explained by "we used a bigger, more accurate draft model" rather than "the RNN + beam search design is inherently better." The paper's claim that ReDrafter "utilizes compute more effectively" (Section 2) requires knowing both the output (speedup, Tokens/Step) *and* the input (draft model size, training cost, inference FLOPs). Without the latter, "more effectively" is unsubstantiated.

A practical consequence: a team evaluating ReDrafter for deployment cannot estimate the memory overhead of adding the draft model to their serving system. Does the RNN + MLP add 50 MB of parameters? 500 MB? 2 GB? For memory-constrained deployments (on-device, edge), this matters enormously — the MLX experiments on M1 Max (Table 2) already show that 13B and 33B models exceed the device's memory, and adding a non-trivial draft model could push even a 7B model over the edge. The paper's silence on draft model size makes memory planning impossible.

The training details gap also prevents assessing whether the distillation advantage (Table 4) is robust or fragile — without knowing how much distillation data was used, whether early stopping was applied, or what learning rate schedule was used, a practitioner cannot estimate the training cost or reliability of reproducing the ~10% gain.

**What evidence exists in the paper.** Very little. Section 3.1 provides the RNN update equations (`s_t = f(U s_{t-1} + W e_t + b)`) and mentions the MLP with skip connections, but no dimensions. Appendix A.1 provides the `dedup_prefix` pseudocode, but no training configuration. The paper's GitHub repository (referenced in the abstract) may contain some of these details, but the paper itself — which is what reviewers and readers evaluate — does not. Section 3.1 explicitly defers model complexity investigations: "In this paper, we opt for a simple recurrent design to model the connections among the shared draft heads, deferring more complex model choices to future investigations." This is reasonable for the RNN architecture choice, but does not excuse omitting the hyperparameters of the architecture actually used.

**Mitigation status.** No mitigation is provided in the paper. The GitHub repository link is given, but the paper should be self-contained enough for readers to understand what was built and how it was trained. This is a standard reproducibility expectation that the paper does not meet. Future work mentions "more advanced distillation techniques" but does not address the documentation gap.

---

### On-Device Evaluation Lacks Baseline Comparisons and Omits Metrics Critical for Mobile Deployment

**The assumption or constraint.** The MLX on-device evaluation (Section 4.2, Table 2, Appendix A.2) demonstrates that ReDrafter achieves 1.32×–2.28× speedup on Apple Silicon relative to autoregressive decoding. The paper claims this demonstrates "its capability to optimize performance in resource-constrained environments" (Section 4.2) and "ReDrafter's viability for on-device use case" (Section 4.2). However, the evaluation has three structural gaps: (1) **no baseline comparison** — Medusa and EAGLE are not evaluated on Apple Silicon, so we cannot assess whether ReDrafter is the *best* on-device speculative decoding method or just *a* working one; (2) **no power or thermal characterization** — on-device deployment is constrained by battery life and thermal throttling, but the paper reports only speedup, not GPU power draw, temperature, or whether the observed speedups are sustainable under extended use; (3) **no latency distribution** — mean TPS is reported, but interactive applications are sensitive to tail latency (p95, p99), and speculative decoding can introduce latency variance because some steps accept many tokens while others accept few.

**The consequence.** A mobile ML team considering ReDrafter for an on-device assistant cannot answer basic deployment questions from the paper's data: Is ReDrafter better than simply using a smaller Medusa draft model on-device? Will sustained use cause the phone to heat up and throttle, eliminating the speedup after a few minutes of conversation? Will users experience occasional multi-second pauses when the draft model produces a sequence that gets fully rejected? The paper's on-device claims are limited to "it works and provides speedup in our measurements" — true but insufficient for production decision-making.

The power/thermal gap is particularly important because speculative decoding explicitly increases total FLOPs (draft model + beam search + LLM verification) to reduce wall-clock time. On a server GPU with unlimited power and active cooling, this is a net win. On a phone with a passive thermal envelope and battery constraints, the increased energy consumption might drain the battery faster or trigger thermal throttling that reduces the effective speedup below the reported numbers (which were presumably measured in a "cold" state). The paper provides no data to address this.

**What evidence exists in the paper.** Table 2 reports TPS and speedup at various beam widths on M1 Max and M2 Ultra. Figure 7 (Appendix A.2) adds beam length sweep heatmaps. The paper notes the sharp TPS drop for Vicuna 33B (1.33 TPS) and acknowledges that "for larger models, compression techniques like quantization may be necessary to achieve acceptable latency" (Section 4.2). Appendix A.2.2 provides implementation lessons (dtype selection, lazy evaluation, JIT compilation), which are practically useful but do not address the missing evaluation dimensions. No baseline comparisons, power measurements, or latency distributions are provided.

**Mitigation status.** The paper partially acknowledges the latency issue for large models (the 33B TPS of 1.33 is described as a scenario where "inference is dominated by memory swapping and data transfer"). The suggestion to use quantization is a forward-looking mitigation. However, the absence of on-device baselines, power characterization, and latency distributions is not acknowledged as a limitation. Future work (Section 5) does not mention on-device evaluation gaps. The paper's GitHub repository may enable others to run baseline comparisons, but this does not replace the need for those comparisons in the paper itself.

---

### The Training Objective Relies on Temperature-0 Distillation Targets, Potentially Making the Draft Model Poorly Calibrated for Non-Greedy Decoding

**The assumption or constraint.** Section 3.5 describes the training procedure: at each position in the training data, the LLM generates the next `T` tokens using temperature 0 ("the LLM generates 5 future tokens at each position of the ground-truth response using a temperature of 0," as clarified in Section 4.3.3). The draft model is then trained to maximize the likelihood of these hard (one-hot) LLM predictions — essentially, it learns to match the LLM's greedy mode, not the LLM's full probability distribution. The KL divergence objective (Equation 1) describes matching distributions, but the empirical loss (Equation 2) uses sampled point estimates at temperature 0, which approximates matching only the mode of the LLM's distribution, not the full distribution.

**The consequence.** This training strategy is well-matched to the evaluation regime (greedy decoding), where the draft model only needs to predict the LLM's top-1 token correctly to achieve acceptance. It is potentially **poorly matched to stochastic decoding**, where acceptance depends on the full distribution. In rejection sampling (the standard speculative decoding verification for temperature > 0), a draft token `x` is accepted with probability `min(1, p_LLM(x) / p_draft(x))`. If the draft model was trained only on the LLM's greedy mode, it may assign inappropriately low probability to tokens the LLM considers plausible (but not top-1), causing those tokens to be rejected even when the LLM would have generated them. Conversely, it may assign inappropriately high probability to tokens the LLM rarely generates, causing them to be accepted too often (distorting the output distribution, though rejection sampling bounds this distortion).

In essence, the draft model may be **overconfident on the LLM's mode and underconfident elsewhere** — a direct consequence of training on one-hot targets rather than soft distributions. This is a standard failure mode of hard-label distillation, well-documented in the knowledge distillation literature. The paper's decision to use temperature 0 for distillation targets is never justified or ablated against higher temperatures that would produce softer, distribution-matching targets.

**What evidence exists in the paper.** None directly. The paper never evaluates with stochastic decoding, so this limitation is entirely unmeasured. The distillation ablation (Table 4) compares only temperature-0 distillation vs. ground-truth training, both evaluated at temperature 0 — it demonstrates that distillation helps in the greedy regime, but provides no information about the stochastic regime. The paper does not report the draft model's calibration (e.g., expected calibration error against the LLM's distribution), which would reveal whether the draft model is distributionally aligned or just mode-aligned.

**Mitigation status.** The paper does not acknowledge this as a limitation. The choice of temperature 0 for distillation targets is presented without justification or ablation. This is a significant oversight because it means ReDrafter's primary reported results may not generalize to the most common deployment scenario (stochastic decoding), and the training procedure may need modification to support that scenario. A straightforward mitigation — using positive temperature when generating distillation targets — is not explored.

## 7. Implications and Future Directions
- Field impact
  - ReDrafter shows that combining recurrence (for accuracy) with dynamic verification (for parallel efficiency) sets a new state-of-the-art speed/acceptance balance for speculative decoding (Table 1). It also demonstrates practicality across execution stacks (PyTorch/CUDA and MLX/Metal), informing device-specific tuning (Figure 7, Table 2–3).
- Near-term applications
  - Latency-sensitive assistants and chat agents (server-side and on-device).
  - Cost-efficient LLM serving (fewer LLM passes per token).
  - On-device inference in constrained environments (Apple Silicon) where memory bandwidth is a bottleneck, with further gains possible via quantization (Section 4.2).
- Research directions
  - Stronger distillation: temperature schedules, sequence-level objectives beyond Eq. (2), and training on sampled, diverse continuations to improve acceptance under non-greedy decoding (Section 5).
  - Variable-length dynamic tree attention: fast tensorized prefix-trie analogs for heterogeneous lengths.
  - More expressive drafters: gated RNNs or lightweight transformers while retaining shared parameters; explore hybrid parallel–recurrent designs to trade off utilization vs. accuracy.
  - Wider LLM coverage: different families, multilingual tasks, and long-context settings; integrate with serving frameworks (the paper notes integration with TensorRT-LLM).
  - Joint optimization with quantization/pruning: co-design drafter, attention masks, and low-bit kernels for mobile-class GPUs.

In short, ReDrafter contributes a principled, implementable path to faster LLM inference: a compact recurrent drafter aligned by distillation, verified efficiently through dynamic tree attention, and tuned carefully to hardware. The method preserves exact LLM outputs while significantly reducing effective decoding steps per token, making it compelling for both datacenter and on-device deployments.
