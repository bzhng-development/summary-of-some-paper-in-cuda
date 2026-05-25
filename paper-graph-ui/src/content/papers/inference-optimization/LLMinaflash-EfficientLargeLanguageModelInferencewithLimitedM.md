# LLM in a flash: Efficient Large Language Model Inference with Limited Memory

**ArXiv:** [2312.11514](https://arxiv.org/abs/2312.11514)

## 🎯 Pitch

'LLM in a flash' presents a novel hardware-aware system that enables large language models—those far exceeding device DRAM capacity—to run efficiently by storing parameters in flash memory and intelligently streaming only the needed weights into DRAM on demand. Through techniques like windowing, row-column bundling, and optimized memory management, the method cuts inference latency by up to 20x compared to naïve approaches and lets devices run models up to 2x larger than available RAM. This breakthrough democratizes on-device LLM inference, making powerful models accessible on everyday hardware without sacrificing performance or privacy.

---

## 1. Executive Summary

This paper tackles the challenge of running LLMs that exceed available DRAM capacity by storing model parameters in flash memory and loading them on demand. Using OPT 6.7B, Falcon 7B, and other models on Apple M1/M2 and NVIDIA hardware, the authors introduce **windowing** — strategically reducing data transfer by reusing neurons activated by recent tokens in a sliding window (keeping the past 5 tokens' active weights in DRAM rather than reloading the full FFN each step) — and **row-column bundling** — co-locating corresponding up-projection columns and down-projection rows in flash so they can be read in larger, contiguous chunks that double the effective chunk size — together enabling models up to 2× the available DRAM capacity with up to 4× speedup on CPU and 20× speedup on GPU versus naive loading approaches (e.g., reducing OPT 6.7B per-token I/O latency from ~2196 ms to ~87 ms on M1 Max when half the model fits in DRAM). The method relies fundamentally on activation sparsity — the FFN layers exhibit over 90% sparsity after ReLU, with a trained low-rank predictor identifying which neurons will fire — establishing that inference from flash is practical only when the base model's activation pattern is sufficiently sparse to make selective loading cheaper than full sequential reads.

## 2. Context and Motivation

### The Core Problem: Running Large Models When DRAM Can't Hold Them

The fundamental problem this paper addresses is brutally simple: **what do you do when your language model is too big to fit in your device's memory?** This isn't a hypothetical edge case — it's the default reality for the vast majority of computing devices in the world. A 7 billion parameter model stored in half-precision (FP16) requires roughly 14 GB just for the weights, without accounting for activations, attention key-value caches, or the operating system's own memory needs. Most smartphones, laptops, and even many desktop computers don't have 14 GB of *available* DRAM to dedicate to a single application, let alone the 28+ GB needed for 13B models or the hundreds of gigabytes required for 70B+ parameter models.

This creates a stark divide in who can actually use state-of-the-art language models. Currently, the standard assumption across the LLM deployment ecosystem is that the entire model must reside in DRAM — either GPU memory (HBM) for GPU-accelerated inference or system DRAM for CPU-based inference. This assumption is baked into every major inference framework: PyTorch, TensorFlow, HuggingFace Transformers, ONNX Runtime — they all expect to `malloc()` enough contiguous memory to hold every parameter simultaneously. If that allocation fails, inference doesn't happen.

The implications of this assumption ripple outward. It means that:
- **On-device deployment** of competitive LLMs is effectively impossible for most consumer hardware. Your smartphone cannot run a 7B model locally, forcing all inference to happen in the cloud with its attendant latency, privacy, and connectivity requirements.
- **Hardware access inequality** is baked into the research ecosystem. Only organizations with access to high-memory GPUs (A100 80GB, H100 80GB) or Apple Silicon Macs with large unified memory pools can experiment with and deploy large models. This concentrates LLM capability in well-resourced institutions.
- **Memory cost dominates inference economics.** DRAM, particularly high-bandwidth GPU memory (HBM), is expensive and supply-constrained. The marginal cost of adding another 16GB of HBM to a GPU often exceeds the cost of many consumer devices entirely. Flash memory, by contrast, is orders of magnitude cheaper per gigabyte and is already present in terabyte-scale quantities on most modern laptops and smartphones.

The paper's motivating observation is that this DRAM-centric paradigm is a **choice, not a necessity**. Flash storage is already present on virtually every computing device, offers capacities at least 10× larger than DRAM at a fraction of the cost, and — critically for LLM inference — only a tiny fraction of the model's parameters are actually needed to process any single token. If we could somehow load *just those needed parameters* from flash on demand, the DRAM requirement would drop from "the entire model" to "the working set of actively used parameters plus overhead." This is the central hypothesis the paper sets out to test and optimize.

### Why This Problem Matters Now (And Why It Wasn't Solved Before)

Three converging trends make this problem urgent and newly tractable:

**1. Model growth has outpaced memory scaling.** The parameter counts of competitive open-source LLMs have roughly doubled every year (from GPT-2's 1.5B in 2019 to Llama 2's 70B in 2023), while consumer device DRAM has grown far more slowly. A flagship smartphone in 2019 had ~4-6GB of RAM; a flagship in 2023 has ~8-12GB. The gap between model size and available memory has widened substantially, and there's no sign of either trend reversing. Flash capacity, meanwhile, has continued its exponential growth — modern phones ship with 128GB–1TB of flash, making it the only storage tier that can actually fit these models.

**2. Activation sparsity makes selective loading feasible.** The critical enabler is that modern LLMs exhibit extreme **activation sparsity** in their feed-forward network (FFN) layers. When a transformer processes a token, the FFN sublayer computes:

$$\text{FFN}(x) = \text{DownProj}(\text{ReLU}(\text{UpProj}(x)))$$

The ReLU activation function naturally zeros out the vast majority of intermediate neurons. For OPT 6.7B, the paper reports **97% sparsity** in FFN intermediate activations — meaning only 3% of the up-projection columns and down-projection rows are actually needed to compute the output for a given token. Falcon 7B, after fine-tuning to use ReLU activation, achieves 95% sparsity. Llama 2, when modified with FATReLU (Song et al., 2024), reaches 90%.

This is not a small optimization opportunity — it's a qualitative shift. If only 3–10% of FFN weights matter per token, then the working set of "actively needed parameters" is potentially an order of magnitude smaller than the full model. The key question becomes: can we predict which parameters will be needed *before* loading them, and can we load them from flash fast enough to not bottleneck inference?

**3. Flash memory characteristics are sufficiently understood.** The paper builds on well-characterized properties of NAND flash storage that are familiar to systems researchers but underappreciated in the ML community. Flash memory exhibits:
- **High sequential read throughput** (over 6 GiB/s on Apple M1 Max for large linear reads of uncached files)
- **Severe throughput degradation for small random reads** (down to ~1 GiB/s or less for sub-64 KiB reads, as shown in Figure 2b)
- **Throughput scaling with read chunk size and parallelism** (larger reads amortize per-request latency overhead; multiple threads exploit controller parallelism)

These characteristics are normally thought of as limitations — "flash is slow for random access" — but the paper reframes them as **design constraints that can be optimized around**. If you structure your data and your access patterns to respect how flash actually works (contiguous chunks, parallel threads), you can achieve throughput within striking distance of what selective weight loading requires. The challenge is aligning the inherently sparse, irregular pattern of "which neurons are active" with the hardware's preference for large, contiguous reads.

### Prior Approaches and Where They Fall Short

The paper identifies several lines of prior work that attempt to address the memory bottleneck for LLM inference, each of which has fundamental limitations that motivate this paper's alternative approach.

#### Model Compression: Quantization and Pruning

Quantization reduces the number of bits per parameter (e.g., from FP16 to INT4), shrinking the model's memory footprint by 4×. Pruning removes parameters entirely by zeroing out weights deemed unimportant. Both techniques are widely studied and deployed.

**Where they fall short:** The paper acknowledges these approaches but identifies a hard ceiling — they reduce the model's size but don't eliminate the requirement that the *entire* (now compressed) model still fits in DRAM. A 7B model quantized to 4 bits still requires ~3.5 GB of DRAM, which may still exceed available memory on many devices. More importantly, quantization and pruning are **orthogonal** to this paper's approach — they reduce what you need to store and load, but they don't address *how* you load it. The paper explicitly notes (Section 1):

> "While it is possible to employ techniques such as quantization to reduce the model size, still, this cannot address the main limitation of loading the entire model into DRAM."

The paper does build on one aspect of pruning-adjacent work: the use of activation sparsity (ReLU-based zeroing) as a dynamic, input-dependent form of "pruning" that happens at inference time rather than being baked into the weights statically. This connects to the sparsification work of Mirzadeh et al. (2023) and Song et al. (2024), which the paper relies on heavily for its base models.

#### Offloading Strategies: FlexGen and Similar Systems

FlexGen (Sheng et al., 2023) is perhaps the most directly comparable prior work. It offloads model weights and the KV cache from GPU memory to DRAM, and from DRAM to flash memory, using a cost model to schedule what gets loaded when. This allows running models larger than GPU memory on a single GPU.

**Where it falls short:** The paper identifies a critical assumption in FlexGen that doesn't hold for on-device scenarios: FlexGen assumes the *entire model* can at least fit in system DRAM, with flash serving as an overflow tier. In the regime this paper targets — where DRAM itself is smaller than the model — FlexGen's strategy of loading weights from flash to DRAM as needed would be "theoretically bound by the slow throughput of flash to DRAM" (Section 6, Appendix E). Specifically, if DRAM can only hold half the model, then every forward pass requires transferring the other half from flash, which at naive sequential read speeds (~6 GB/s) means:

$$\text{transfer time} = \frac{\text{half model size}}{\text{flash bandwidth}} \approx \frac{6.7 \text{ GB}}{6.1 \text{ GB/s}} \approx 1.1 \text{ seconds per token}$$

This is the "naive" baseline that the paper benchmarks (2196 ms per token for OPT 6.7B on M1 Max). FlexGen's scheduling optimizations help when you're juggling GPU and DRAM, but they don't fundamentally reduce the volume of data that must cross the flash→DRAM boundary. The paper's key insight is that you need to reduce that volume — and the only way to do so is to exploit activation sparsity to load *only the needed weights* rather than the full half-model.

#### Selective Weight Loading: Deja Vu and Predictor-Based Approaches

Deja Vu (Liu et al., 2023b) is the most conceptually related prior work. It trains small predictors to identify which FFN neurons will be activated by a given input, then loads only the corresponding weight rows for computation. This is exactly the paradigm this paper adopts: predict → load selectively → compute.

**Where it falls short:** Deja Vu's key limitation is that it still assumes **GPU memory** is large enough to hold the predictors and attention weights, with the selective loading happening from GPU memory (HBM) rather than from flash. The paper builds directly on Deja Vu's predictor architecture but extends it to the fundamentally more constrained setting where selective loading must happen from flash memory — a storage medium with bandwidth roughly 100× lower than GPU HBM and very different access pattern constraints.

This extension is non-trivial. Loading 3% of FFN weights from GPU memory (bandwidth ~1-2 TB/s) adds negligible latency. Loading 3% of FFN weights from flash (bandwidth ~1-6 GB/s for random reads) can still be a bottleneck if not carefully optimized. The entire "hardware-informed" framing of this paper — the cost model, the chunk-size optimization, the bundling strategy — arises precisely because flash, unlike GPU memory, punishes small random reads so severely.

#### Selective Execution: Mixture of Experts and Conditional Computation

Mixture of Experts (MoE) architectures naturally load only a subset of FFN "experts" per token, creating a built-in sparsity pattern that maps cleanly to selective loading. EdgeMoE (Yi et al., 2023) specifically targets on-device MoE inference.

**Where it falls short:** MoE is an architectural choice, not a technique applicable to existing dense models. The paper's approach works with standard dense transformer architectures (OPT, Falcon, Llama, Persimmon, Phi) by exploiting the *emergent* sparsity that arises from ReLU activations, without requiring MoE-specific training or architecture modifications. The paper explicitly notes (Appendix E) that MoE models "can leverage our method for enabling larger models on the device," positioning its approach as complementary rather than competing.

#### Hardware-Centric Approaches: Processing-in-Memory, Custom Accelerators

There is a rich body of work on hardware solutions: processing-in-memory (PIM) that moves compute closer to storage, custom ASIC accelerators with large on-chip SRAM, and specialized memory hierarchies. These include computed RAM (Gao et al., 2022), neural cache architectures (Meswani et al., 2015), and flash-based inference systems like HotPot (Shao et al., 2022).

**Where they fall short:** These approaches require custom hardware that doesn't exist in deployed devices. The paper's stated goal is to enable inference on *existing* personal devices — the MacBooks, smartphones, and desktop GPUs that people already own. This constrains the solution space to algorithmic and data layout optimizations that work within the capabilities of commodity flash storage controllers, operating system I/O stacks, and standard deep learning frameworks. The "hardware-informed" aspect of the paper is about understanding and adapting to existing hardware, not designing new hardware.

### How This Paper Positions Itself

The paper carves out a specific and previously unexplored niche in the inference optimization landscape:

**The setting:** DRAM capacity is smaller than the model size. Not just "smaller than the full working set including activations and KV cache" — fundamentally smaller than the weights alone. This is the regime where the standard "load everything into DRAM" assumption breaks down and no amount of intra-DRAM optimization can help.

**The strategy:** Store the full model in flash memory (where capacity is abundant and cheap), but load only the dynamically needed subset of weights into DRAM on demand. This is fundamentally a **data transfer minimization** problem: the primary cost is not FLOPs (computation) but bytes moved across the flash→DRAM boundary.

**The intellectual framework:** All optimizations are derived from a three-component cost model:

$$\text{Latency} = \underbrace{\text{I/O cost}}_{\text{flash → DRAM transfer}} + \underbrace{\text{memory management}}_{\text{DRAM reallocation overhead}} + \underbrace{\text{compute cost}}_{\text{FLOPs for forward pass}}$$

The paper's techniques map cleanly to these components:
- **Windowing** reduces the I/O cost by exploiting temporal locality in neuron activations (Section 3.1)
- **Row-column bundling** reduces the I/O cost by increasing effective transfer throughput (Section 3.2)
- **Preallocated memory management** reduces the memory management overhead (Section 3.3)

Compute cost is explicitly treated as orthogonal — the paper "focus[es] on optimizing flash memory interactions and memory management" rather than computational efficiency (Section 3).

**The key dependency:** Everything depends on **activation sparsity**, which determines how much data transfer can be avoided. Without sparsity (or a reliable predictor of which weights will be needed), the approach collapses to the naive baseline of loading half the model per token. The paper is explicit that this is not a general solution for all LLMs — it works for models whose FFN activations are sufficiently sparse that the cost of predicting and selectively loading active neurons is less than the cost of loading everything sequentially.

This positioning creates a clear decision boundary: if your model has >90% FFN activation sparsity and your DRAM is 40-60% of model size, the techniques described here can deliver practical speedups. If your model is dense (e.g., uses GELU or Swish activations without sparsification fine-tuning) or your DRAM is so tiny that even the sparse working set doesn't fit, you need different approaches (quantization, model distillation, or cloud offloading).

The paper's contribution is thus not a single technique but an **integrated system** that combines sparsity prediction (building on Deja Vu), hardware-aware data layout (building on flash storage systems research), and memory management (building on systems programming practice) into a coherent inference pipeline that collectively makes flash-resident LLM inference practical for the first time.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper builds an **inference engine** that runs large language models whose full parameter set cannot fit in the available DRAM by storing the complete model weights in flash memory and loading only the dynamically needed subset per token. The core problem it solves is the DRAM capacity wall: when you have (say) 7GB of available memory but a 14GB model, traditional inference frameworks simply fail because they require the entire model to be resident in memory. The solution's "shape" is a three-part optimization over a hardware-informed cost model — reduce what you load (via sparsity prediction and windowing), load it faster (via chunk-size optimization through row-column bundling), and manage what's in memory efficiently (via preallocated, pointer-based data structures) — where all three parts are necessary because flash memory's throughput characteristics punish small random reads so severely that you cannot simply load the sparse weights naively and expect acceptable latency.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components, arranged in a pipeline that processes one token at a time:

1. **Low-Rank Predictor Network** — a small learned model attached to each transformer layer that takes the current attention output as input and predicts which FFN intermediate neurons will produce positive (non-zero) ReLU activations. Its output is a binary mask over intermediate neurons. Responsibility: avoid loading up-projection columns and down-projection rows for neurons that will be zeroed by ReLU anyway.

2. **Sliding Window Cache** — a DRAM-resident cache containing the weight rows for neurons that were active during the most recent `k` tokens (typically `k = 4` or `5`). Responsibility: exploit the temporal locality of neuron activations across adjacent tokens so that only the *difference* between the current token's active set and the previous token's active set needs to be loaded from flash, rather than the entire active set each time.

3. **Bundled Flash Storage Layout** — the on-disk arrangement of FFN weights where, for each intermediate neuron `i`, the `i`-th column of the up-projection matrix is physically stored adjacent to the `i`-th row of the down-projection matrix. Responsibility: double the size of each flash read chunk (from `d_model × num_bytes` to `2 × d_model × num_bytes`) without increasing the volume of data transferred, because the up-projection column and down-projection row for the same neuron are always needed together — if neuron `i` fires, both its column and its row are required for the FFN computation.

4. **Preallocated Memory Manager** — a DRAM data structure (pointer array, matrix buffer, scalar array, usage counter, and last-k tracking) that avoids repeated `malloc`/`realloc`/`memcpy` operations as new neurons are loaded and old neurons are evicted. Responsibility: minimize the "memory management" component of the latency cost model by replacing dynamic allocation with a fixed-size buffer and pointer-based rewrites.

5. **Multithreaded Flash Reader** — a parallel I/O subsystem that reads required weight chunks from flash using 32 concurrent threads. Responsibility: amortize the per-request "latency to first byte" overhead by issuing multiple reads simultaneously, and exploit the inherent parallelism of the flash controller to approach the sequential read throughput ceiling even when reads are logically scattered across the flash device.

Information flows through the system as follows for each generated token:

**Step 1 (Attention):** The full attention weights (which are permanently DRAM-resident, constituting roughly one-third of the model) compute the attention output from the current input embedding and the KV cache.

**Step 2 (Prediction):** The low-rank predictor takes the attention output and produces a binary mask over intermediate FFN neurons — this mask says "neuron 42 will fire, neuron 517 will fire, neuron 3 will not fire, ..."

**Step 3 (Window update):** The system compares the predicted active set for the current token with the active sets of the previous `k` tokens stored in the sliding window cache. Neurons needed by the current token but not present in the cache are identified for loading. Neurons present in the cache but not needed by any of the last `k` tokens are identified for eviction.

**Step 4 (Flash read):** The memory manager issues parallel read requests to flash for each new neuron's bundled weight chunk (up-projection column concatenated with down-projection row). The 32-thread reader dispatches these reads. Each read fetches `2 × d_model × num_bytes` bytes from flash into a preallocated DRAM buffer.

**Step 5 (Memory update):** The memory manager deletes evicted neurons by overwriting their slots with the last active entries (maintaining contiguous occupation), then appends newly loaded neuron weights to the end of the buffer.

**Step 6 (Compute):** The FFN forward pass executes using only the rows present in the DRAM buffer. The up-projection uses the first half of each row (`matrix[:num_rows, :d_model]`); the down-projection uses the transposed second half (`matrix[:num_rows, d_model:].transpose()`). The sparse intermediate activation vector is computed, ReLU is applied (which further zeroes most entries — the predictor already anticipated this), and the output is produced.

**Step 7 (Next token):** The FFN output feeds into the next transformer layer's attention module, and the process repeats for the next token.

At the very start of a sequence, the sliding window cache is empty. The first token loads all its predicted active neurons fully from flash. Subsequent tokens typically need only a small fraction of new neurons because adjacent tokens share most of their active neurons — this is the "incremental transfer" optimization enabled by windowing.

### 3.3 Roadmap for the Deep Dive

- **First, the inference cost model (3.4.1):** This formalizes what "optimal" means — latency decomposed into I/O, memory management, and compute components — and establishes why reducing data transfer volume and increasing chunk size are the two primary levers. Without this model, the design choices in the subsequent sections lack motivation.

- **Second, the low-rank predictor (3.4.2):** This is the gateway component that enables everything else — if you can't predict which neurons will fire, you can't do selective loading. I explain what the predictor is architecturally, how it's trained (loss function, data, hyperparameters), why it uses the attention output rather than the previous FFN output (a key difference from Deja Vu), and the accuracy-sparsity trade-off it navigates. The predictor is covered first because all subsequent mechanisms assume the existence of a reliable sparsity prediction.

- **Third, the sliding window technique (3.4.3):** This builds on the predictor to exploit temporal locality across tokens. I explain the mathematical formulation of aggregated neuron usage `s_agg(k)`, the incremental loading principle `s_agg(k+1) - s_agg(k)`, why the slope of aggregated usage decreases with window size (Figure 4a), and the memory-latency trade-off that governs window size selection. This is the "reduce data transfer" pillar.

- **Fourth, row-column bundling and chunk-size optimization (3.4.4):** This addresses the "increase transfer throughput" pillar. I explain how co-locating corresponding up-projection columns and down-projection rows on flash achieves 2× larger chunks without loading extra data, why this matters given flash's random-read throughput curve (Figure 2b), and the negative result on co-activation-based bundling (Appendix D) that clarifies why this simple 2× bundling is the effective strategy rather than more sophisticated neuron grouping.

- **Fifth, the memory management system (3.4.5):** This covers the "efficient management of loaded data" pillar. I walk through the preallocated data structure (pointer, matrix, bias, num_used, last_k_active), the deletion and insertion operations that avoid reallocation and copying, and why the commutative property of FFN intermediate outputs — neuron order doesn't matter — enables this simplified layout. Without this, the memory management overhead (25% of FFN data rewritten per token) would cancel the I/O savings.

- **Sixth, the selective persistence strategy and end-to-end DRAM budget (3.4.6):** This explains which weights are permanently DRAM-resident (embeddings, attention weights, predictors) versus flash-resident with on-demand loading (FFN weights), and works through a concrete memory budget for OPT 6.7B to show how everything fits within the ~50% DRAM constraint.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is an **inference systems paper** whose core idea is that the hardware characteristics of flash memory — specifically, the severe throughput penalty for small random reads and the throughput scaling with both chunk size and parallelism — fundamentally change what constitutes an efficient LLM inference strategy, and that by co-designing the data layout, the memory management, and the sparsity exploitation to match these characteristics, you can achieve practical inference speeds even when DRAM holds only half the model.

---

#### 3.4.1 The Inference Cost Model

The paper frames all optimization decisions through a latency decomposition. For each token generated, the total inference latency `L` is the sum of three terms:

$$L = L_{\text{I/O}} + L_{\text{mem}} + L_{\text{compute}}$$

where `L_I/O` is the time spent reading weight data from flash into DRAM, `L_mem` is the time spent managing the DRAM-resident data structures (deleting evicted neurons, inserting newly loaded ones), and `L_compute` is the time spent performing the actual matrix multiplications and attention computations.

**What this formalization captures:** the end-to-end wall-clock time per token, decomposed by subsystem. `L_I/O` is determined by the volume of data transferred and the effective throughput of the flash storage stack for the access pattern used. `L_mem` is determined by the number of DRAM read/write operations needed to maintain the working set. `L_compute` is determined by the model architecture and the hardware's floating-point throughput.

**Why this decomposition matters:** the three terms have different scaling behaviors and respond to different optimizations. `L_compute` is largely orthogonal to the flash-loading strategy — the paper explicitly sets it aside ("our focus is not on optimizing the compute") and concentrates on `L_I/O` and `L_mem`. Within `L_I/O`, the paper identifies two independent levers:

1. **Data volume:** `L_I/O` scales roughly linearly with the number of bytes transferred from flash. Reducing the number of parameters loaded per token (via sparsity prediction and windowing) directly reduces `L_I/O`.

2. **Effective throughput:** `L_I/O` also depends on the achieved bandwidth. For flash, this is strongly dependent on read chunk size. The relationship is non-linear: throughput degrades severely for sub-64KB reads because the fixed "latency to first byte" overhead dominates the transfer time for small reads. Increasing chunk size (via row-column bundling) increases throughput without necessarily increasing data volume, hence reducing `L_I/O`.

The third term, `L_mem`, becomes non-negligible when the working set changes significantly between tokens. If deleting old neurons and inserting new ones requires rewriting large contiguous arrays (as would happen with naive `realloc` + `memcpy` approaches), `L_mem` can rival `L_I/O`. The preallocated memory management strategy (Section 3.4.5) targets this term directly.

The paper does not provide a single closed-form equation for `L_I/O` in terms of data volume and chunk size — it treats the relationship empirically, benchmarking actual throughput for different chunk sizes and thread counts (Figure 2b). The key empirical fact that drives all design decisions is:

> "For smaller blocks, a substantial part of the overall read time is spent waiting for data transfer to begin. This is often referred to as latency to first byte. This latency reduces the overall throughput of each read operation considerably because the overall measured throughput has to take into account not just the speed of transfer once it begins, but the latency before it begins as well, which penalizes small reads."

This has a counterintuitive consequence that the paper exploits:

> "in some scenarios, it will be worthwhile to read more than needed (but in larger chunks) and then discard, rather than only reading strictly the necessary parts but in smaller chunks"

This principle underlies the row-column bundling strategy: by physically co-locating the up-projection column and down-projection row for each neuron, the system reads 2× the strictly-needed data per neuron (since both the column and row are needed anyway when the neuron fires), but does so in a chunk that is 2× larger, achieving higher throughput. The net effect is lower latency because the throughput gain more than offsets the marginal extra data.

---

#### 3.4.2 Low-Rank Activation Predictor

**What it is.** For each transformer layer's FFN sublayer, a small auxiliary neural network takes the current token's attention output as input and predicts which intermediate FFN neurons will produce a positive (non-zero) output after the ReLU activation. This is a binary classification problem per neuron: given the attention output vector `h ∈ ℝ^(d_model)`, predict a binary mask `m ∈ {0,1}^(d_ffn)` where `m_i = 1` if the `i`-th intermediate neuron's pre-activation value (the result of `UpProj(h)` before ReLU) is positive, and `m_i = 0` if it is zero or negative.

**Architecture.** The predictor is a low-rank linear model. It computes:

$$\hat{m} = \text{sigmoid}(W_2 \cdot \text{ReLU}(W_1 \cdot h))$$

where `W_1 ∈ ℝ^(r × d_model)` and `W_2 ∈ ℝ^(d_ffn × r)` are the predictor's weight matrices, and `r` is the predictor's rank — a hyperparameter that controls the predictor's capacity and computational cost. The sigmoid output produces per-neuron probabilities; during inference, a threshold (typically 0.5) converts these to a binary mask.

**Input choice: why attention output and not previous FFN output.** This is a specific design decision that the paper explicitly contrasts with prior work. Deja Vu (Liu et al., 2023b) uses the *previous layer's FFN output* as input to the predictor for the current layer's FFN. The paper instead uses the *current layer's attention output*. The authors state:

> "In contrast to their work, our predictor needs only the output of the current layer's attention module and not the previous layer's FFN module. We have observed that postponing the prediction to the current layer is sufficient for hardware-aware weight-loading algorithm design but leads to more accurate outcomes due to deferred inputs."

The rationale is that the current layer's attention output contains more up-to-date information about the token's representation than the previous layer's FFN output, which is "stale" by one sublayer. Using the current attention output also simplifies the hardware pipeline: the attention computation must happen anyway (since attention weights are permanently DRAM-resident), so the predictor can run in parallel with or immediately after attention without waiting for the previous layer's FFN to load and execute.

**Training procedure.** The predictor is trained per-layer, independently for each transformer block. The training data is constructed as follows:

1. For 10,000 samples from the C4 training dataset, run the full (unmodified) model in inference mode and record, for each layer, the attention output `h` and the ground-truth binary mask `m` (computed by running the full up-projection and checking which pre-activation values are positive before ReLU).

2. Train the predictor to minimize a balanced binary cross-entropy loss:

$$\mathcal{L} = -\frac{1}{d_{\text{ffn}}} \sum_{i=1}^{d_{\text{ffn}}} \left[ w_{\text{pos}} \cdot m_i \log(\hat{m}_i) + w_{\text{neg}} \cdot (1 - m_i) \log(1 - \hat{m}_i) \right]$$

where `m_i ∈ {0, 1}` is the ground-truth activation of neuron `i`, `m̂_i ∈ [0, 1]` is the predictor's estimated probability, and `w_pos`, `w_neg` are class weights that balance the contribution of positive and negative examples.

**What this loss computes:** a per-neuron binary cross-entropy, summed across all `d_ffn` intermediate neurons, with class reweighting to handle extreme class imbalance. Since typically 3–10% of neurons are active (positive class), an unweighted loss would train the predictor to simply predict "always zero" and achieve 90–97% accuracy. The balanced loss forces the predictor to care about the rare positive examples.

**Why balanced loss:** the paper states "We used a balanced loss over negative and positive samples of each layer." This ensures the predictor learns to identify the small minority of active neurons rather than collapsing to the trivial all-zeros solution. Without balancing, the predictor would have high accuracy but zero recall — it would never predict any neuron as active, making it useless for selective loading.

**Training hyperparameters.** The paper provides specifics: "10000 samples from the C4 training dataset to do the training for 2 epochs. It took 4 hours on an A100 GPU to train each predictor." The rank `r` varies by layer and by model. For OPT 6.7B: "For the initial 28 layers of the OPT 6.7B model, we train predictors with a rank of `r = 128`. To reduce the occurrence of false negatives, the final four layers employ predictors with a higher rank of `r = 1024`." For Falcon 7B: `r = 256` for initial layers, `r = 1152` for the last four. For Persimmon 8B: `r = 256` for the first 32 layers, `r = 1152` for the last four. The higher rank in later layers reflects the empirical observation (Figure 9d) that "later layers have more active neurons" and thus require more predictor capacity to maintain low false-negative rates.

**Why per-layer, why these ranks:** training one predictor per layer is necessary because the sparsity pattern varies substantially across layer depth (Figure 9a-c). Early layers are sparser, later layers are denser. Using a single global predictor would fail to capture this variation. The specific rank choices (128, 256, 1024, 1152) are empirically determined to balance predictor accuracy against predictor size — larger ranks improve accuracy but consume more DRAM (since predictors are permanently resident) and add more computational overhead.

**Predictor overhead.** The paper quantifies: "The average rank of predictors in the OPT-6.7B is 240, this will result in less than 2.4% of non-embedding weights and FLOPs. In M1 Max CPU experiments this was comprising 2.75% and in RTX GPU it was 4.8% of inference time which is negligible." For Falcon 7B: "predictors take 4% model size and CPU computation." For Persimmon: "2.85% of inference time on CPU." The overhead is small enough that the net speedup from selective loading dominates.

**Accuracy-quality trade-off.** The predictor is not perfect. As Figure 3a shows, the predictor "accurately identifies most activated neurons, while occasionally misidentifying inactive ones with values near zero." There are two types of errors:

- **False positives:** neurons predicted active but actually zero. These cause unnecessary weight loading (loading up-projection columns and down-projection rows that will be multiplied by zero), wasting I/O bandwidth but not affecting output correctness — multiplying by a zero row produces zero contribution regardless.

- **False negatives:** neurons predicted inactive but actually non-zero. These cause the model to skip neurons that would have contributed to the output, potentially degrading accuracy.

The paper reports that OPT 6.7B predictors achieve "an average of 5% false negatives and 7% false positives." Critically, the false negatives tend to have near-zero activations: "these false negatives, being close to zero, do not significantly alter the final output when they are excluded." This is an empirical property of the ReLU activation distribution — most active neurons have small magnitudes, and the largest-magnitude activations are rarely missed by the predictor.

**Threshold tuning.** The sigmoid threshold for converting predictor probabilities to binary masks is not always 0.5. For Persimmon 8B, the paper notes: "we changed the sigmoid threshold to 0.7" because the model's sparsity is lower than OPT and Falcon. A higher threshold means the predictor is more conservative: it predicts fewer neurons as active, loading less data but risking more false negatives. Table 4 in Appendix B shows the accuracy-latency trade-off: threshold 0.5 gives the highest zero-shot metrics, while 0.7 improves efficiency at a small accuracy cost.

**Verification of accuracy impact.** Table 1 (main paper) and Table 4 (Appendix B) demonstrate that predictor use does not meaningfully degrade model quality:

- OPT 6.7B: Arc Easy drops from 66.1 to 66.2, Arc Challenge unchanged at 30.6, HellaSwag drops from 50.3 to 49.8 (Table 1).
- Falcon 7B (relufied, with predictors `r = 1152/256`): Arc Easy 72.35, Arc Challenge 36.35, HellaSwag 53.16 — close to the base relufied model (Table 4).
- Persimmon 8B (with predictors, threshold 0.7): Arc Easy 66.30, Arc Challenge 34.40, HellaSwag 52.70 (Table 4).
- Phi-2: MMLU drops from 54.3 (relufied + distilled) to 52.0 with predictors (Figure 10b), a non-trivial but acceptable degradation given the 2.35× speedup.
- Llama 2 7B (sparsified): MMLU 38.96 for sparsified model, 38.63 after predictors — essentially unchanged.

The paper is transparent about the trade-off: "An interesting future direction of this work is improving the accuracy of the predictors to be able to load fewer neurons." The current predictors deliberately "over-predict" (load ~3× the truly active neurons on average) to keep false-negative rates low, trading some I/O efficiency for output quality preservation.

---

#### 3.4.3 The Sliding Window Technique

**The intuition.** The predictor tells us which neurons are active for the *current* token. Without windowing, we would load all of those neurons from flash for every token, even if 90% of them were also active for the previous token. This is wasteful — adjacent tokens in a sequence tend to share most of their active neurons, a form of temporal locality. The sliding window technique exploits this by keeping the active neurons of the most recent `k` tokens cached in DRAM, so that only the *newly activated* neurons (those active for the current token but not cached) need to be loaded from flash.

**Mathematical formulation.** Let `s_agg(k)` denote the average number of distinct neurons that are active in at least one of the last `k` tokens — the **aggregated neuron usage** over a window of size `k`. For a single token ( `k = 1` ), `s_agg(1)` is the average number of active neurons per token, which (given 90%+ sparsity) is roughly 3–10% of `d_ffn`. As `k` increases, `s_agg(k)` grows because tokens farther apart activate increasingly different sets of neurons — but it grows **sublinearly**. The paper's key observation, visualized in Figure 4a, is:

> "the slope of aggregated neuron usage is decreasing"

Mathematically, the **incremental transfer** needed when processing the `(k+1)`-th token while maintaining a window of the preceding `k` tokens is:

$$\Delta(k) = s_{\text{agg}}(k+1) - s_{\text{agg}}(k)$$

**What `Δ(k)` represents:** the number of new neurons that must be loaded from flash to DRAM when advancing the sliding window by one token — neurons that are active for token `t+1` but were not active for any of tokens `t-k+1` through `t`.

**Why `Δ(k)` decreases with `k`:** as the window grows, the cache already contains neurons from a longer history of tokens, making it increasingly likely that a neuron needed by the new token is already present. The first few tokens in a window add a large number of distinct neurons (the "new topic" neurons), but each subsequent token adds fewer and fewer new neurons (the "continuing topic" neurons). This is the temporal locality property that windowing exploits.

For example, in the tenth layer of Falcon 7B (Figure 4a), `s_agg(1)` is roughly 3% of `d_ffn`, `s_agg(5)` is roughly 15%, `s_agg(30)` is roughly 42%. The incremental transfer `s_agg(5) - s_agg(4)` is about 2%, while `s_agg(30) - s_agg(29)` is about 0.3%. Larger windows thus dramatically reduce the per-token I/O load — but at the cost of higher DRAM usage for the cache itself.

**The memory-latency trade-off.** The window size `k` is the primary tunable parameter that controls the balance between two competing costs:

- **DRAM usage:** The cache stores `s_agg(k) × 2 × d_model × num_bytes` bytes (each cached neuron stores both its up-projection column and down-projection row, hence the factor of 2). Larger `k` means storing more neurons in DRAM, which directly competes with the available memory budget.

- **I/O latency:** The per-token flash load volume is approximately `Δ(k) × 2 × d_model × num_bytes`. Larger `k` means smaller `Δ(k)`, hence lower I/O latency per token.

The paper sets `k` to the largest value that fits within the available DRAM budget. For OPT 6.7B with ~50% of model size available in DRAM: "Utilizing a windowing method with `k = 4` in the OPT 6.7B model significantly reduces the necessity for fresh data loading. Using active neurons of predictor would require about 10% of the DRAM memory capacity on average; however, with our method, it drops to 2.4%."

**Concrete example: OPT 6.7B windowing.** With `k = 4` (maintaining a cache of neurons active in the last 5 tokens — the current token plus 4 predecessors), the per-token incremental load drops from ~10% of FFN neurons (no windowing) to ~2.4% (with windowing). This is a 4.2× reduction in data transfer volume. The cost is increased DRAM for the cache: the window occupies "24% of the Feed Forward Network" in DRAM, up from (presumably) 10% without caching. The total DRAM budget is detailed in Section 3.4.6.

**Why k = 4 or 5 across most experiments:** this is the value that fits within the ~50% total DRAM budget while providing significant I/O reduction. The paper explores varying `k` in Figure 7's memory-latency trade-off analysis: as `k` increases, more of the model is resident in DRAM, latency decreases, but DRAM usage increases. The sweet spot depends on the specific hardware's DRAM capacity relative to model size.

**Window eviction policy.** The paper uses a simple policy: neurons are evicted from the cache when they are not among the predicted active neurons for any of the last `k` tokens. This is tracked efficiently using the `last_k_active` field in the memory management data structure (Section 3.4.5). The eviction is exact — a neuron is kept if and only if it appears in the active set of at least one token in the current window.

**Why not an LRU or frequency-based policy:** the paper doesn't discuss alternatives, but the choice is well-motivated by the problem structure. Tokens are sequential and the window size `k` is small (4–5). An exact "active in recent `k` tokens" policy is computationally cheap (one integer comparison per cached neuron per token), deterministic, and directly matched to the autoregressive generation pattern where recent tokens are the most predictive of future activations.

**The "first few tokens" effect.** The paper notes an important transient: "the flash latency for the first few tokens is higher since the allocated memory in DRAM is empty and needs to be filled in with neurons and for first few tokens we need more data transfer." At sequence start, the cache is empty, so `s_agg(1)`, `s_agg(2)`, ... are all loaded fully. The I/O latency is front-loaded. After `k` tokens, the cache reaches steady state, and the per-token load drops to `Δ(k)`. This means the technique is most beneficial for long generation sequences where the steady-state savings dominate over the initial cache-filling cost.

**Long-generation stability.** The paper verifies (Figure 8) that the per-token flash latency does not increase over 1000-token generations on GPU for OPT 6.7B, confirming that the windowing scheme maintains its efficiency indefinitely. This also rules out thermal throttling concerns: "it is possible that for longer generation of tokens, the SSD enable thermal throttling and lower the performance. However, Figure 8 shows that this is not the case."

**Nucleus sampling compatibility.** The autoregressive sliding window depends on temporal locality — the assumption that adjacent tokens share many active neurons. This holds for greedy decoding but could break for stochastic sampling methods that produce more diverse outputs. The paper tests this explicitly: "We found out this is not the case either for long token generations. Nucleus sampling doesn't lead to lower performance in long generation in neither cpu or gpu." The temporal locality appears robust to the specific decoding strategy.

---

#### 3.4.4 Row-Column Bundling and Chunk-Size Optimization

**The problem: small reads kill throughput.** Flash memory achieves its maximum bandwidth (over 6 GiB/s on M1 Max, as benchmarked for 1GiB linear reads) only for large, sequential reads. For small random reads, throughput degrades dramatically. Figure 2b quantifies this: at 4KB chunk size with 2 threads, throughput is roughly 200 MB/s; at 64KB with 32 threads, it approaches 6000 MB/s (the sequential read upper bound). The paper explicitly explains the mechanism:

> "For smaller blocks, a substantial part of the overall read time is spent waiting for data transfer to begin. This is often referred to as latency to first byte."

Each flash read request has a fixed overhead — the operating system must set up the I/O, the driver must dispatch the command, the flash controller must locate the requested blocks — before any data bytes are transferred. For large reads, this fixed overhead is amortized over many bytes; for small reads, it dominates the total time.

**The insight: corresponding columns and rows are always needed together.** In the FFN layer of a transformer, the computation for a given intermediate neuron `i` uses:
- The `i`-th column of the up-projection matrix `W_up[:, i]` (to compute neuron `i`'s pre-activation)
- The `i`-th row of the down-projection matrix `W_down[i, :]` (to project neuron `i`'s post-ReLU activation back to `d_model`)

These are **co-dependent**: if neuron `i` fires (pre-activation > 0), both `W_up[:, i]` and `W_down[i, :]` are needed for the computation. If neuron `i` doesn't fire, neither is needed (the ReLU output is zero, so `W_down[i, :]` is multiplied by zero and contributes nothing).

**The bundling strategy.** Instead of storing `W_up` and `W_down` as two separate matrices on flash, the paper **interleaves** them by neuron index. For each neuron `i`, the column `W_up[:, i]` (size `d_model × num_bytes`) is stored immediately followed by the row `W_down[i, :]` (also size `d_model × num_bytes`). When neuron `i` is predicted active, the system reads one contiguous chunk of size `2 × d_model × num_bytes` from flash, obtaining both the column and the row in a single I/O operation.

**Why this doubles chunk size without increasing data volume.** The data transferred per active neuron is unchanged — you need both the column and the row regardless. But without bundling, you would issue two separate reads (one for the column, one for the row), each of size `d_model × num_bytes`. With bundling, you issue one read of size `2 × d_model × num_bytes`. The chunk size doubles, moving it from the low-throughput regime toward the higher-throughput regime of Figure 2b, while the total bytes transferred remain identical.

**Concrete example for OPT 6.7B.** With `d_model = 4096` and FP32 (`num_bytes = 4`): each column or row is `4096 × 4 = 16,384` bytes = 16 KiB. Without bundling, each active neuron requires two 16 KiB reads. With bundling, it requires one 32 KiB read. From Figure 2b's data, a 16 KiB read with moderate thread counts achieves perhaps 500–1000 MB/s, while a 32 KiB read achieves perhaps 1000–2000 MB/s — roughly double the throughput. This throughput doubling directly halves `L_I/O` for the same data volume.

**Quantifying the benefit.** Table 2 provides the empirical breakdown for OPT 6.7B on M1 Max:

- With predictor and windowing but **no bundling**: transfer 0.2 GB at throughput 1.25 GB/s → I/O latency 164 ms
- With predictor, windowing, and **bundling**: transfer 0.2 GB at throughput 2.25 GB/s → I/O latency 87 ms

Bundling increases throughput from 1.25 GiB/s to 2.25 GiB/s (1.8× improvement) and reduces I/O latency from 164 ms to 87 ms (1.88× speedup). The throughput improvement is less than the theoretical 2× because reads are not purely sequential even with bundling — the active neuron indices are scattered across the `d_ffn` dimension, so the bundled reads are still random-access in flash space, just with larger per-request sizes.

**The counterintuitive principle: read more, not less.** The paper explicitly articulates a design principle that might seem paradoxical:

> "in some scenarios, it will be worthwhile to read more than needed (but in larger chunks) and then discard, rather than only reading strictly the necessary parts but in smaller chunks"

This applies here because bundling doesn't actually read "more than needed" — it reads exactly what's needed (both column and row) but in one larger request rather than two smaller ones. But the principle generalizes: if additional co-located data can be read with negligible marginal cost (due to the chunk size effect), it may be cheaper to over-read and discard than to issue separate small reads. The paper explores a more aggressive version of this idea in Appendix D (co-activation bundling, which attempted to bundle neurons that frequently fire together — but this proved counterproductive for reasons discussed below).

**Threading for additional throughput.** The bundling strategy is deployed alongside multithreaded reads. The paper states:

> "To optimize data loading from flash memory, our system employs reads parallelized over 32 threads. This multithreaded approach is intended to both better amortize latency to the first byte by not waiting for each read sequentially, and maximize read throughput by reading multiple streams at once (Figure 2b)."

The 32-thread count is not arbitrary — it's chosen based on the throughput scaling shown in Figure 2b, where 32 threads achieve near-peak throughput for chunk sizes above 32 KiB. With bundling producing 32 KiB chunks and 32 parallel readers, the system operates near the right edge of Figure 2b's throughput curve.

**The negative result: co-activation bundling (Appendix D).** The paper explored a more sophisticated bundling strategy: grouping neurons that frequently co-activate (fire together on the same tokens) and storing their weights contiguously on flash. The hypothesis was that when one neuron in a co-activation group fires, its "buddies" likely fire too, so reading the entire group together would be efficient.

The analysis of co-activation patterns (Figure 12) revealed a power-law distribution: for each neuron, its most frequently co-activated partner (the "closest friend") co-activates with it 95–100% of the time. The 4th closest friend co-activates about 86% of the time, and the 8th about 75%. This seemed promising — bundling each neuron with its top few friends could amplify the chunk size by 4–8× with only modest "wasted" reads.

However, the results were negative:

> "Unfortunately, this resulted in loading highly active neurons multiple times and the bundling worked against our original intention. It means the neurons that are very active are the 'closest friends' of almost everyone."

The problem is asymmetry: neuron A might have neuron B as its closest friend (and B co-activates with A 95% of the time), but B might be a universally active neuron that is the closest friend of hundreds of other neurons. Grouping A with B leads to B being loaded whenever A fires (which is efficient) but also whenever any of B's other "friends" fire (which leads to B being loaded many times for tokens where it's not needed). Since universally active neurons consume cache space and I/O bandwidth, the redundant loading outweighs the chunk-size benefit.

The paper deliberately includes this negative result "as we believe it may inspire future research on effective neuron bundling and its utilization for efficient inference." This is a rare instance of a systems paper publishing a failed approach, and it serves to clarify why the simple 2× bundling (one neuron's column + its own row) is the right granularity — it bundles exactly co-dependent data with no risk of redundant loading.

**Storage overhead of bundling.** The bundling scheme requires reorganizing the flash storage layout from two separate matrices to one interleaved array. This is a one-time preprocessing step — the model weights are stored in bundled format on flash and never need to be rearranged at inference time. The storage size is unchanged: `2 × d_model × d_ffn × num_bytes` bytes total (for up-projection columns and down-projection rows combined), regardless of layout.

---

#### 3.4.5 Preallocated Memory Management

**The problem: dynamic memory allocation is expensive at inference time.** As the sliding window advances, the set of neurons cached in DRAM changes: old neurons (no longer active in the last `k` tokens) must be removed, and new neurons (predicted active for the current token but not cached) must be inserted. The naive approach — `realloc` the weight buffer to accommodate the new size, `memcpy` the surviving weights to their new positions, then `free` the old buffer — incurs substantial overhead. The paper quantifies this:

> "When introducing data for new neurons, reallocating the matrix and appending new matrices can lead to significant overhead due to the need for rewriting existing neuron data in DRAM. This is particularly costly when a substantial portion (approximately 25%) of the Feed-Forward Networks (FFNs) in DRAM needs to be rewritten."

If 25% of the cached FFN weights must be rewritten per token, and the cached FFN occupies (say) 25% of the model's total DRAM budget, this could easily add tens of milliseconds of pure memory copying per token — potentially rivaling or exceeding the flash I/O time that the sparsity and windowing optimizations worked so hard to reduce.

**The solution: preallocated fixed-size buffer with pointer-based indirection.** Instead of dynamically resizing arrays, the memory manager preallocates a buffer large enough to hold the maximum expected number of cached neurons for the given window size. This maximum is determined empirically:

> "The matrix for the i-th layer is pre-allocated with a size of `Req_i × 2d_model`, where `Req_i` denotes the maximum number of neurons required for the specified window size in a subset of C4 validation set."

The factor of `2d_model` accounts for the bundled storage: each row in the buffer stores one neuron's up-projection column (size `d_model`) concatenated with its down-projection row (also size `d_model`).

**Data structure.** Each FFN layer maintains a single management structure with the following fields (Figure 6):

- `matrix`: a preallocated 2D buffer of shape `(Req_i, 2 × d_model)`, stored in row-major order. Each row `j` holds a complete bundled (column, row) pair for one cached neuron.

- `pointer`: an integer array of length `Req_i` where `pointer[j]` stores the original neuron index (0 through `d_ffn - 1`) corresponding to row `j` of the matrix. This indirection is what allows the buffer to maintain contiguous occupation of rows while the set of cached neurons changes — the same neuron index might appear at different row positions over time.

- `bias`: a scalar array of length `Req_i` where `bias[j]` stores the up-projection bias value for the neuron at row `j`.

- `num_rows`: an integer tracking how many of the `Req_i` rows are currently occupied with valid neuron data. Initially 0, grows as the cache fills.

- `last_k_active`: an array tracking, for each cached neuron, the most recent token index at which it was predicted active. Used to identify neurons eligible for eviction.

**Why this works: neuron order doesn't matter for FFN computation.** The key insight that makes this simplified layout possible is a mathematical property of the FFN's intermediate representation. The FFN computation is:

$$\text{FFN}(x) = \sum_{i \in \text{active}} W_{\text{down}}[i, :] \cdot \text{ReLU}(W_{\text{up}}[:, i]^T x + b_i)$$

This is a sum over active neurons. Because addition is commutative, the order of summation does not affect the result. Therefore, the rows in the DRAM buffer can be in any order — the system doesn't need to maintain the original neuron indices 0, 1, 2, ... in sequence. It can place newly loaded neurons at any empty slot (in practice, always at the end for efficiency) and delete neurons by overwriting their slots with other valid data, without preserving order.

**Deletion operation (Figure 6, steps 1-2).** When a neuron is no longer active in the last `k` tokens, it must be removed from the cache to make room for new neurons. The deletion proceeds as follows:

1. Identify evicted neurons by comparing `last_k_active` values with the current token index. Neurons whose last activation was more than `k` tokens ago are candidates.

2. For each evicted neuron at row position `j` in the matrix, **overwrite** that row with the data from the last occupied row (position `num_rows - 1`). This is a fixed-size copy of `2 × d_model` elements.

3. Copy the corresponding `pointer[num_rows - 1]` and `bias[num_rows - 1]` to positions `j`.

4. Decrement `num_rows` by the number of evicted neurons.

The cost: "For `O(c)` neurons to be deleted, a memory rewrite of the order `O(c × d_model)` is required." This is linear in the number of evicted neurons, not in the total cache size. Critically, it avoids shifting all subsequent rows down (which would be `O(num_rows × d_model)`).

**Insertion operation (Figure 6, step 3).** When new neurons must be loaded:

1. Issue flash read requests for the bundled (column, row) data of each new neuron.

2. Place the loaded data into `matrix[num_rows : num_rows + num_new]` — appending to the end of the currently occupied region.

3. Set `pointer[num_rows + j]` to the original neuron index for each new neuron `j`.

4. Set `bias[num_rows + j]` to the corresponding up-projection bias value.

5. Increment `num_rows` by `num_new`.

No existing data is moved during insertion. No reallocation occurs because the buffer was preallocated to `Req_i`, which is guaranteed sufficient for the window size in use.

**Inference operation (Figure 6, step not numbered).** During the actual FFN forward pass, the system uses only the first `num_rows` rows of the matrix:

- Up-projection: `matrix[:num_rows, :d_model]` — the first `d_model` columns of each occupied row.
- Down-projection: `matrix[:num_rows, d_model:].transpose()` — the last `d_model` columns of each occupied row, transposed so that neuron dimension becomes the input and `d_model` becomes the output.

The system does not need to look up original neuron indices during computation because the bundled layout already pairs each up-projection column with its corresponding down-projection row, and the commutative property means any row ordering produces identical results.

**Why preallocation to `Req_i` is safe.** The paper validates that `Req_i` is an upper bound on the number of cached neurons for the chosen window size by empirically measuring the maximum cache occupancy across a subset of the C4 validation set. If the window size `k` is chosen such that the steady-state cache occupancy never exceeds the preallocated capacity, no reallocation is ever needed. The paper's DRAM budget calculations (Section 3.4.6) ensure this condition is met.

**Quantified benefit.** The paper doesn't report a direct "with vs. without memory management" ablation in the main text, but the latency breakdowns in Table 3 show that `L_mem` (the "Mem" column) ranges from 8 ms (M2 Ultra) to 92 ms (Falcon 7B on CPU) to 155 ms (Persimmon 8B on CPU). These are non-trivial fractions of total latency. Without the preallocation and pointer-based overwrite strategy, these numbers would be substantially higher (potentially 2–4× by the paper's estimate of ~25% data rewriting overhead).

---

#### 3.4.6 Selective Persistence Strategy and End-to-End DRAM Budget

**Which weights are permanently DRAM-resident?** The paper makes a strategic decision about which components of the transformer stay in DRAM at all times versus being fetched from flash on demand:

1. **Embeddings** (token embedding table): Always in DRAM. Size: `vocab_size × d_model`. For OPT 6.7B with `vocab_size = 50272` and `d_model = 4096` in FP32: `50272 × 4096 × 4 ≈ 824 MB`. This is roughly 3% of the model.

2. **Attention weights** (all `Q`, `K`, `V`, `O` projection matrices across all layers): Always in DRAM. The paper states these "constitute approximately one-third of the model's size." For OPT 6.7B's total 13.4 GB, this is about 4.3 GB.

3. **Predictor networks** (all per-layer low-rank predictors): Always in DRAM. For OPT 6.7B, "The Predictor accounts for 1.25% of the model size" — about 168 MB.

4. **Feed-forward network (FFN) weights** (up-projection and down-projection across all layers): Stored in flash, loaded on demand. These constitute "approximately two-thirds of the model's size" or about 8.9 GB for OPT 6.7B. Only a dynamically selected subset is in DRAM at any time, governed by the sliding window cache.

**Why attention weights are kept resident.** The paper gives a clear justification: "Keeping attention weights, which constitute approximately one-third of the model's size, in memory, allows for more efficient computation and quicker access, thereby enhancing inference performance without the need for full model loading." Attention weights are accessed for *every* token and are not amenable to sparsity-based selective loading in the same way FFN weights are — the attention mechanism computes dense `QK^T` products that involve all attention heads regardless of token content. Keeping them in DRAM avoids the complexity and I/O cost of partial attention weight loading, which the paper deems not worthwhile given that attention is only one-third of the model.

**Why predictors are kept resident.** The predictors are small (1.25–4% of model size depending on rank choices) and must be executed *before* the FFN weights can be loaded (since they determine which weights to load). Keeping them in DRAM eliminates a chicken-and-egg problem and adds negligible memory overhead.

**Concrete DRAM budget for OPT 6.7B at ~50% constraint.** The paper works through the arithmetic in Appendix C.1:

- Embeddings: 3% of model size
- Attention weights: 32.3% of model size
- Predictor: 1.25% of model size
- FFN cached (24% of FFN, which is 64.62% of model): 0.24 × 64.62 = 15.5% of model size
- **Total DRAM:** 3.0 + 32.3 + 1.25 + 15.5 = 52.05% of model size

This fits within the ~50% constraint (which the paper treats as approximate — 52% is "approximately half"). The FFN cache fraction (24%) is determined by the window size `k = 4` (storing neurons from the last 5 tokens). At steady state, the cache requires `s_agg(5)` neurons' worth of storage, which empirical measurement shows averages 24% of total FFN weight volume.

**What if DRAM is tighter or looser?** The paper explores this in Section 4.3 (Figure 7) and in the ablation for smaller models. For Phi-2 (2.7B parameters), where the model is already small relative to typical DRAM, the constraint is relaxed to 65% (i.e., 65% of model size available in DRAM). For scenarios with even less DRAM, the window size `k` must be reduced, which increases `Δ(k)` (the incremental load per token) and thus increases latency, but allows operation under tighter memory budgets. The paper frames this as a continuous trade-off:

> "By increasing the window size, we increase the percentage of model parameters that we keep in DRAM. As a result, we need to bring fewer parameters, and hence the latency can be reduced at the cost of using higher DRAM as shown in Figure 7."

**The hybrid baseline.** The paper defines a "hybrid" approach as an intermediate baseline: keep half the model in DRAM permanently (the attention weights and whatever else fits), and load the other half from flash on every forward pass, without using sparsity prediction or windowing. This is the theoretically optimal approach *if you don't exploit activation sparsity* — you load exactly the weights that aren't in DRAM, which is half the model. The I/O latency for this baseline (Table 2, row 2) is 1090 ms for OPT 6.7B on M1 Max (6.7 GB transferred at 6.10 GB/s). This is the starting point against which all sparsity-based optimizations are measured.

**How the full system achieves 87 ms I/O instead of 1090 ms.** The chain of improvements from Table 2:

| Method | Data Transferred | Throughput | I/O Latency |
|--------|-----------------|------------|-------------|
| Naive (load all) | 13.4 GB | 6.10 GB/s | 2196 ms |
| Hybrid (half loaded) | 6.7 GB | 6.10 GB/s | 1090 ms |
| + Predictor | 0.9 GB | 1.25 GB/s | 738 ms |
| + Windowing | 0.2 GB | 1.25 GB/s | 164 ms |
| + Bundling | 0.2 GB | 2.25 GB/s | 87 ms |

The predictor reduces data volume by 6.7× (from 6.7 GB to 0.9 GB), but throughput drops from 6.10 GB/s to 1.25 GB/s because the reads are now small and scattered. Windowing further reduces volume by 4.5× (to 0.2 GB), but throughput remains low. Bundling doubles the chunk size without changing data volume, boosting throughput to 2.25 GB/s, which cuts I/O latency in half. The combination of reduced volume and increased throughput achieves a **25× reduction in I/O latency** (2196 → 87 ms) compared to naive loading.

**Why the throughput drops with sparsity.** The dense reads (naive, hybrid) are large, contiguous transfers that achieve the flash device's full sequential read bandwidth (6.10 GB/s). The sparse reads (predictor, windowing) are small, scattered reads — each active neuron requires a separate random-access read to fetch its bundled column-row pair from its position in the interleaved layout. These random reads achieve much lower throughput (Figure 2b) because of the latency-to-first-byte penalty. Bundling partially mitigates this by making each random read larger, but cannot fully recover sequential bandwidth because the reads are still randomly distributed across the flash device's address space. The throughput achieved (2.25 GB/s) represents the saturation point for 32 KiB random reads with 32 parallel threads on the M1 Max's flash subsystem.

## 4. Key Insights and Innovations

### Innovation 1: Reframing Flash Memory from "Slow Storage" to "Contiguous-Read Optimized Transfer Medium"

The dominant assumption in both the ML inference and systems communities has been that flash memory is fundamentally unsuitable for LLM inference because it's "slow." This belief is well-founded in the numbers: flash bandwidth (~6 GB/s sequential) is roughly 100× lower than GPU HBM bandwidth (~1-2 TB/s), and random-read throughput can be another 10-50× worse than sequential. The natural conclusion — drawn implicitly by virtually all prior inference systems — is that flash belongs in a deep storage tier, accessed only for initial model loading, never during per-token inference. FlexGen (Sheng et al., 2023) offloads to flash as an overflow from DRAM but treats flash I/O as a cost to be minimized through scheduling, not optimized through data layout.

This paper's foundational conceptual move is to **invert the question**. Instead of asking "how can we minimize expensive flash accesses?", it asks "given that flash has specific throughput characteristics (Figure 2b), how should we restructure our data and access patterns to make flash-resident inference practical?" This is a shift from treating flash characteristics as constraints to treating them as **design parameters** that should govern algorithm and data layout choices.

What makes this intellectually distinctive is that it reveals the conventional wisdom about flash being "slow" as an oversimplification. Flash is slow *for small random reads*, but it is reasonably fast for large contiguous reads — and the throughput gap between these two regimes spans nearly two orders of magnitude on the same hardware. The paper's core technical insight — that you can achieve practical inference speeds if you simultaneously reduce data volume (via sparsity) AND increase chunk size (via bundling) AND parallelize reads — follows directly from this reframing. None of the three techniques alone would suffice; it's the hardware-informed co-design that makes the system work.

The counterintuitive corollary that "it will be worthwhile to read more than needed (but in larger chunks) and then discard" (Section 3.2) crystallizes this reframing. Prior work operating under the "flash is slow" assumption would never intentionally over-read, because the goal was always to minimize bytes transferred. This paper recognizes that bytes transferred is the wrong objective — latency is what matters, and latency is a function of both bytes and chunk sizes, with a nonlinear relationship that can make larger reads cheaper than smaller ones even if they transfer more data. This is systems thinking applied to ML inference in a way that was previously absent from the literature.

### Innovation 2: Sliding Window Caching as a Temporal Locality Exploitation Strategy for Sparse Activations

The observation that adjacent tokens share active neurons is not, by itself, novel — temporal locality is a fundamental property of autoregressive generation that has been exploited in other contexts (KV caching for attention, for instance). What makes this paper's sliding window technique intellectually distinctive is the **quantitative characterization** of how aggregated neuron usage `s_agg(k)` grows with window size, and the recognition that the *slope* of this growth — not the absolute value — determines incremental loading cost.

Prior work on selective weight loading (Deja Vu, Liu et al., 2023b) treated each token independently: predict active neurons for token `t`, load all of them, compute, discard. There was no notion of a cache across tokens. This made selective loading viable from GPU memory (where bandwidth is high enough that loading 3-10% of FFN weights per token is acceptable) but completely infeasible from flash, where even 3% of FFN weights per token (roughly 0.2 GB for a 7B model at 90% sparsity) would require ~160 ms at random-read throughput, far too slow for interactive use.

The windowing technique transforms the problem from "load 3% per token" to "load ~0.5% per token" (the incremental transfer `Δ(k)`), a 6× further reduction beyond sparsity alone. This is not an incremental improvement — it's the difference between flash-resident inference being marginally practical and being clearly practical. The mathematical formulation `Δ(k) = s_agg(k+1) - s_agg(k)` captures why: the aggregated usage curve has decreasing slope (Figure 4a), meaning that as the window grows, each additional token adds progressively fewer *new* neurons to the working set. The first token in a new topic activates many distinct neurons; the tenth token on the same topic activates almost entirely neurons already seen.

The significance extends beyond raw performance. This analysis establishes that **the cacheability of sparse activations is what makes flash-resident inference viable**, not sparsity alone. If neuron activations were independent across tokens (no temporal locality), even 97% sparsity wouldn't suffice — you'd still need to load 3% of FFN weights from flash per token at random-read speeds, yielding ~150+ ms I/O latency. The paper's empirical finding that `Δ(5)` is roughly 2.4% for OPT 6.7B (vs. ~10% without windowing) is the quantitative evidence that temporal locality is strong enough to bridge the gap between flash bandwidth and inference latency requirements.

### Innovation 3: The Difficulty-Aware Memory-Latency Trade-off as a Tunable Deployment Axis

Prior work on model compression and efficient inference typically treats memory usage as a hard constraint — you either fit in the available DRAM or you don't. Quantization, pruning, and distillation all produce a single "compressed model" with fixed memory requirements. If it doesn't fit, you need a different technique or a different model.

This paper introduces a fundamentally different paradigm: **memory usage is a continuous tunable parameter** that can be adjusted at deployment time via the window size `k`, with a smooth and predictable trade-off against inference latency. Figure 7 demonstrates this empirically: as you allocate more DRAM to the FFN cache (by increasing `k`), per-token latency decreases because fewer new neurons need to be loaded from flash. There is no cliff, no binary "works/doesn't work" threshold — just a continuous curve that a deployer can navigate based on their specific hardware constraints and latency requirements.

This is conceptually significant because it transforms the deployment problem from "find a model that fits" to "find the operating point on the memory-latency curve that satisfies your requirements." A device with 40% of model size available in DRAM can still run the model — just with higher latency. A device with 60% available DRAM can run it faster. The same model, the same flash storage layout, the same predictor weights — only the window size changes. This is a level of deployment flexibility that no prior inference optimization technique provides.

The innovation is not the existence of the trade-off (any cache-based system has one) but the **explicit characterization and operator-level control** of it. The window size `k` is a single integer parameter that directly governs both DRAM usage and I/O latency, with predictable, monotonic effects on each. This makes it a practical knob for deployment engineers rather than a research hyperparameter that requires retraining or architecture changes. The paper doesn't fully explore the automation of this trade-off (e.g., dynamically adjusting `k` based on available memory or latency targets), but it establishes the framework that makes such automation possible.

### Innovation 4: The Predictor Architecture Shift from "Previous FFN Output" to "Current Attention Output" as a Latency-Hiding Strategy

At first glance, the choice of predictor input — current layer's attention output vs. previous layer's FFN output — seems like a minor implementation detail. Deja Vu (Liu et al., 2023b) used the previous FFN output; this paper uses the current attention output. The paper's stated justification is that "postponing the prediction to the current layer is sufficient for hardware-aware weight-loading algorithm design but leads to more accurate outcomes due to deferred inputs."

The deeper significance, however, is that this choice is not primarily about accuracy — it's about **enabling predictor execution to overlap with computation that must happen anyway**, thereby hiding predictor latency. The attention computation is always performed (attention weights are permanently DRAM-resident), so running the predictor on the attention output means the prediction happens in parallel with or immediately after attention, without waiting for the previous layer's FFN to load its weights from flash, compute, and produce its output. In the Deja Vu design, the predictor for layer `i` depends on layer `i-1`'s FFN output, creating a serial dependency: load FFN weights → compute FFN → run predictor for next layer → load next layer's FFN weights. In this paper's design, the dependency chain is: compute attention → run predictor in parallel with attention output → load FFN weights, which shortens the critical path.

This is a subtle but meaningful contribution to the **systems-ML co-design** philosophy that runs through the paper. The predictor architecture is chosen not just for its statistical properties (accuracy, parameter count) but for how it slots into the overall inference pipeline to minimize serial dependencies. The paper doesn't make a big theoretical claim about this, but the design choice reflects a sophistication about hardware-aware ML that distinguishes it from prior work where the predictor was treated purely as a statistical component.

The evidence for this claim is partially circumstantial — the paper reports that predictor overhead is "2.75% of inference time on CPU" and "4.8% on GPU," which is negligible, but doesn't provide a head-to-head comparison of current-attention vs. previous-FFN predictor architectures in terms of end-to-end latency. The accuracy comparison (claiming "more accurate outcomes due to deferred inputs") is asserted but not quantitatively benchmarked against Deja Vu's approach. This makes the innovation more of a **design insight** than a rigorously validated finding, but it's a design insight with clear systems-level motivation that future work can build on.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All latency benchmarks use a small subset of the C4 validation dataset: "We take the first 128 tokens of each example as the prompt, and generate 256 new tokens" (Section 4.1). For predictor training, 10,000 samples from the C4 training dataset are used. Zero-shot accuracy evaluations use standard benchmarks: ARC Easy, ARC Challenge, HellaSwag, MMLU, and Wikitext2 perplexity.

- **Base model(s).** The primary models are OPT 6.7B (Zhang et al., 2022b), a sparsified Falcon 7B (Mirzadeh et al., 2023) — the original Falcon uses GELU activation; it is fine-tuned with ReLU to achieve 95% sparsity — Phi-2 2.7B (Gunasekar et al., 2023), Persimmon 8B (Elsen et al., 2023), and a sparsified Llama 2 7B using FATReLU (Song et al., 2024). These span multiple model families (OPT, Falcon, Phi, Persimmon, Llama) and sizes (2.7B to 8B), covering different activation functions and sparsity patterns. The authors state the techniques are "mostly independent of architecture" (Section 4.1).

- **Metrics.** The primary metric is **per-token inference latency (milliseconds)**, decomposed into three components: I/O latency (time spent reading weights from flash to DRAM), memory management latency (time spent updating DRAM data structures — deleting evicted neurons, inserting new ones), and compute latency (time for matrix multiplications and attention). For I/O-specific analysis, **throughput (GB/s)** and **data transferred (GB)** are reported separately. Model quality is measured via zero-shot accuracy on standard benchmarks and perplexity on Wikitext2.

- **Baselines.**
  - **Naive:** Load the entire model from flash on every forward pass. The paper notes "the most efficient theoretical baseline involves loading half of the model size from the flash memory into DRAM" (Section 4.1) because with ~50% DRAM available, at least half the model must be transferred at least once per forward pass.
  - **Hybrid:** Keep half the model permanently in DRAM (embeddings and attention weights), load the remaining half (all FFN weights) from flash on demand for every token — no sparsity prediction, no windowing, no bundling. The paper describes this as a secondary baseline that represents "the theoretically optimal approach *if you don't exploit activation sparsity*" and states "we are not aware of any other method that can surpass this theoretical I/O efficiency" without leveraging sparsity.

- **Generation budget / compute accounting.** All comparisons are at the per-token level with identical compute workloads (same model, same sequence, same number of FLOPs). The budget is implicitly: one forward pass through the full transformer per generated token. The reported latencies aggregate across 256 generated tokens per sequence. The paper does not use a FLOPs-based accounting because compute is treated as orthogonal — the optimization target is flash I/O and memory management, not computation reduction. For speculative decoding experiments, the draft model generates λ = 4 candidate tokens which the full model verifies; the cost model for window management in this setting is discussed qualitatively.

- **Cross-validation / statistical protocol.** Predictor training uses 10,000 C4 samples; hyperparameters (ranks per layer, sigmoid thresholds) are tuned per model based on observed sparsity patterns and false-negative rates. The preallocation size `Req_i` for each layer's DRAM buffer is determined by measuring maximum cache occupancy on a subset of the C4 validation set. No formal cross-validation across prompt datasets is reported for latency measurements. Latency values in Table 3 are accompanied by standard deviations in Appendix Table 5, computed across tokens within the generation: e.g., OPT 6.7B "All" on CPU reports I/O latency of 104.90 (± 18.46) ms, memory management 57.79 (± 9.63) ms, compute 506.50 (± 17.33) ms, total 669.20 (± 39.74) ms. For accuracy evaluations (Table 1, Table 4, Figure 10), standard zero-shot benchmarks are used with their established evaluation protocols. Flash throughput benchmarks are deliberately pessimized by disabling OS caching: "On macOS and iOS, we employ the F_NOCACHE flag with the fcntl() function, while on Linux, we use DirectIO. Additionally, on macOS, we clear any resident buffers before initiating the benchmark using the purge command" (Appendix C). This provides conservative lower-bound throughput measurements.

### Main Quantitative Results

#### End-to-End Inference Latency Across Hardware Platforms

Table 3 presents the headline results: end-to-end per-token latency for each model under each method (Naive, Hybrid, All) on each hardware backend. The "All" configuration — predictor + windowing + bundling + optimized memory management — consistently achieves dramatic speedups:

- **OPT 6.7B on CPU (M1 Max):** Naive latency 3182 ms (I/O: 2196, Mem: 0, Compute: 986). All: 669 ms (I/O: 105, Mem: 58, Compute: 506) — a **4.75× speedup**. The I/O component alone is reduced by 20.9×.

- **OPT 6.7B on GPU (Apple Metal, M1 Max):** Naive 2389 ms. All: 565 ms — a **4.2× speedup**.

- **OPT 6.7B on GPU (Apple Metal, M2 Ultra):** Naive 2270 ms. All: 305 ms — a **7.4× speedup**. The M2 Ultra achieves substantially lower compute (271 ms vs. 438 ms on M1 Max) and memory management (8 ms vs. 35 ms) latency, reflecting the newer hardware's capabilities.

- **OPT 6.7B on GPU (NVIDIA RTX 4090):** Naive 2218 ms. All: 84 ms — a **26.4× speedup**. This is the most dramatic result and underpins the paper's "up to 20-25x in GPU" claim (the 20-25× range appears in Section 7; 26.4× is the specific number for OPT). The compute component drops to 20 ms (vs. 986 on CPU), reflecting the RTX 4090's vastly superior floating-point throughput.

- **Falcon 7B on CPU:** Naive 3095 ms, Hybrid 1947 ms, All 706 ms — a **4.4× speedup over Naive**, 2.8× over Hybrid.

- **Persimmon 8B on CPU:** Naive 3806 ms, Hybrid 2495 ms, All 1041 ms (note: Table 3 says 1041 ms; Appendix Table 5 reports 1090.08 (± 79.08) ms with the note that Persimmon's sparsity is lower and sigmoid threshold was raised to 0.7, increasing latency slightly) — a **3.7× speedup over Naive**.

- **Phi-2 2.7B on CPU:** Naive 1287 ms, Hybrid 711 ms, All 546 ms — a **2.4× speedup over Naive**. The smaller relative gain reflects Phi-2's lower sparsity rates and the fact that the baseline is already faster (smaller model, less data to transfer).

- **Llama 2 7B on CPU:** Naive 3095 ms, Hybrid 1903 ms, All 994 ms — a **3.1× speedup over Naive**.

#### Decomposition of I/O Latency Improvements

Table 2 provides the critical step-by-step decomposition showing how each technique contributes to I/O latency reduction for OPT 6.7B on M1 Max:

- **Naive (entire model from flash, no DRAM caching):** 13.4 GB transferred at 6.10 GB/s → 2196 ms. This represents loading the full 13.4 GB model from flash every forward pass.

- **Hybrid (half model in DRAM, half from flash):** 6.7 GB transferred at 6.10 GB/s → 1090 ms. The data volume is halved (only FFN weights loaded), and throughput remains at sequential-read peak (6.10 GB/s) because the reads are large and contiguous.

- **+ Predictor (sparsity-based selective loading, no windowing, no bundling):** 6.7 GB DRAM used, 0.9 GB transferred at 1.25 GB/s → 738 ms. Data volume drops by 7.4× (from 6.7 GB to 0.9 GB) because only ~3% of FFN weights are loaded per token. However, throughput collapses from 6.10 GB/s to 1.25 GB/s — a 4.9× reduction — because reads are now small, scattered, random accesses. The net I/O latency improvement is modest: 1090 → 738 ms (1.48×), because the throughput degradation partially offsets the volume reduction.

- **+ Windowing (predictor + sliding window, no bundling):** 6.5 GB DRAM used, 0.2 GB transferred at 1.25 GB/s → 164 ms. Data volume drops another 4.5× (0.9 → 0.2 GB) because only ~2.4% incremental new neurons are loaded per token (Δ(4) vs. the full active set). Throughput remains at 1.25 GB/s. I/O latency drops from 738 ms to 164 ms (4.5× improvement), directly proportional to the data volume reduction.

- **+ Bundling (predictor + windowing + bundling):** 6.5 GB DRAM used, 0.2 GB transferred at 2.25 GB/s → 87 ms. Data volume unchanged (0.2 GB), but throughput increases 1.8× (1.25 → 2.25 GB/s) because each read is now 32 KiB instead of 16 KiB, moving it into a higher-throughput regime on the flash device's throughput-vs-chunk-size curve. I/O latency drops from 164 ms to 87 ms (1.88× improvement).

The combined effect from Hybrid to All: 1090 → 87 ms, a **12.5× I/O latency reduction**. The DRAM usage remains roughly constant (~6.5 GB for the cache + permanent weights vs. 6.7 GB for the hybrid baseline's full FFN), meaning the improvement comes entirely from reducing transfer volume and increasing throughput, not from using more DRAM.

#### Memory-Latency Trade-off

Figure 7 (described in Section 4.3) characterizes the relationship between DRAM allocation and per-token latency for OPT 6.7B on the GPU machine. The x-axis is the percentage of model parameters resident in DRAM, ranging from approximately 35% to 80%. The y-axis is per-token latency (ms), stacked into three components: Load From Flash, Memory Management, and Compute.

The key finding: as DRAM allocation increases from ~35% to ~80%, total latency decreases from roughly 40 ms to roughly 10 ms. The "Load From Flash" component dominates at low DRAM allocations (~30 ms at 35%) and nearly vanishes at high allocations (~2 ms at 80%). The Memory Management and Compute components remain relatively constant across the range (Compute hovering around 2-3 ms, Memory Management around 2-5 ms). This empirically validates the modeling assumption that I/O is the dominant cost and that the memory-latency trade-off is smooth and predictable — there is no cliff, no threshold below which inference becomes impractically slow. The paper frames this as enabling deployment flexibility: "By increasing the window size, we increase the percentage of model parameters that we keep in DRAM. As a result, we need to bring fewer parameters, and hence the latency can be reduced at the cost of using higher DRAM" (Section 4.3).

#### Long-Generation Stability and Sampling Robustness

Figure 8 addresses a potential concern: whether flash I/O latency degrades over long sequences due to SSD thermal throttling or other effects. For OPT 6.7B on GPU generating 1000 tokens, the per-token flash latency (labeled "DRAM → Flash Latency") is plotted against generation length for three configurations: CPU with Nucleus sampling, GPU with Nucleus sampling, and GPU with Greedy decoding.

The result: per-token flash latency is approximately flat across all 1000 tokens for all three configurations. The paper reports that "the average flash latency doesn't increase as we go further in generation." There is a visible spike in the first ~50 tokens (latency starting at roughly 300-600 ms and dropping to steady-state values of ~50-100 ms), which the paper explains: "the flash latency for the first few tokens is higher since the allocated memory in DRAM is empty and needs to be filled in with neurons and for first few tokens we need more data transfer." After the cache reaches steady state (~50-100 tokens), latency stabilizes and remains constant through token 1000. The paper also confirms that Nucleus sampling does not degrade performance: "Nucleus sampling doesn't lead to lower performance in long generation in neither cpu or gpu."

#### Speculative Decoding Integration

Table 3 includes a "Speculative" row for OPT 6.7B on GPU: I/O 38.5 ms, Memory Management 9.5 ms, Compute 12 ms, Total 60 ms — compared to the "All" configuration's 84 ms. This is a **1.4× speedup** from adding speculative decoding on top of the flash-loading system. The paper provides implementation details: "Given λ tokens from the draft model, the big model verifies them and will keep a window of size k for each layer... We used λ = 4 and were able to improve the speed of decoding by 1.4x as shown in table 5, this is close to the original 1.58x speedup of speculative decoding" (Section 5.2). The speculative decoding result is significant because it demonstrates that the windowing mechanism can be adapted to the batch-verification pattern of speculative decoding — the window is updated with multiple tokens at once rather than one at a time, and the neuron retention policy must account for the possibility that most draft tokens get rejected.

#### Accuracy Preservation Under Predictor Use

Table 1 (main text) and Table 4 (Appendix B) quantify how predictor-based selective loading affects model quality. The key findings:

- **OPT 6.7B** (Table 1): Arc Easy: 66.1 (original) → 66.2 (with predictors). Arc Challenge: 30.6 → 30.6 (unchanged). HellaSwag: 50.3 → 49.8. The drop is negligible across all three benchmarks.

- **Falcon 7B** (Table 4): The relufied (ReLU-fine-tuned) model achieves Arc Easy 72.52, Arc Challenge 38.23, HellaSwag 54.17. With predictors at varying ranks, the best configuration (rank 1152 for last 4 layers, 256 for others, threshold 0.5) achieves Arc Easy 72.35, Arc Challenge 36.35, HellaSwag 53.16 — a small degradation, particularly in Arc Challenge (-1.88 points). Lower predictor ranks (128 for all layers) cause more significant drops: Arc Easy 70.20, Arc Challenge 35.41, HellaSwag 50.74. This validates the paper's design choice to use larger predictors (higher rank) in later layers.

- **Persimmon 8B** (Table 4): With threshold 0.5: Arc Easy 67.26, Arc Challenge 33.87, HellaSwag 50.51 — close to the original 67.80/34.64/50.70. Increasing the threshold to 0.7 (which reduces data load but increases false-negative risk) yields Arc Easy 66.30, Arc Challenge 34.40, HellaSwag 52.70 — HellaSwag actually improves slightly, while Arc Easy and Challenge remain comparable. This trade-off space is explored across thresholds 0.5, 0.55, 0.60, 0.65, 0.70 in Table 4, showing a gentle degradation curve rather than a sharp cliff. The paper uses threshold 0.7 for Persimmon specifically because "Persimmon's sparsity is less than OPT and Falcon."

- **Phi-2** (Table 4, Figure 10b): After relufication + distillation: MMLU drops from 57.0 (original Phi-2) to 54.3. After predictor addition: MMLU drops to 52.0. This is a 2.3-point drop from the relufied model and 5.0 points from the original — the largest accuracy impact among all models tested. Varying the sigmoid threshold from 0.40 to 0.55 changes zero-shot metrics only modestly (Arc Easy 79.96→78.90, Arc Challenge 49.57→47.90, HellaSwag 53.50→52.75 at threshold 0.55). The paper attributes this drop to "the relufication process due to lower quality data" (Figure 10b caption), noting that "using distillation can improve the results."

- **Llama 2 7B (sparsified)** (Section C.5): MMLU: 41.8 (original Llama 2) → 38.96 (sparsified via FATReLU) → 38.63 (with predictors). The additional drop from predictors alone is 0.33 points — essentially noise-level.

- **Perplexity** (Figure 11): OPT 6.7B and Persimmon show "a slight drop in perplexity" after predictor use. Phi-2 shows "more drop" — consistent with the larger MMLU degradation observed.

### Ablation Studies and Robustness Checks

- **Step-by-step technique decomposition (Table 2):** Each component of the system — predictor, windowing, bundling — is evaluated in isolation by adding it to the previous configuration. This ablation demonstrates that no single technique is sufficient: the predictor alone provides only 1.48× I/O improvement over Hybrid because throughput collapses when reads become sparse; windowing provides 4.5× improvement over predictor-only by reducing data volume; bundling provides an additional 1.88× by restoring some of the lost throughput. The full stack is necessary for the reported 12.5× I/O latency reduction.

- **Predictor rank and threshold tuning (Table 4):** For Falcon 7B, comparing predictor ranks of 128 vs. 1152 for the last 4 layers: the higher rank recovers 1.31 Arc Easy points, 0.94 Arc Challenge points, and 2.42 HellaSwag points — a substantial accuracy improvement that justifies the increased DRAM cost. For Persimmon 8B, comparing thresholds from 0.5 to 0.70 shows that accuracy drops by at most ~1.5 points across tasks, while enabling significant data transfer reduction. The paper identifies that without larger predictors in the last layers, MMLU drops significantly for Persimmon (Figure 10a): "If we use larger predictors in the last 4 layers MMLU wouldn't drop a lot in Persimmon. This wouldn't be the case if all the predictors had a rank of 256."

- **Co-activation-based bundling (negative result, Appendix D):** The paper attempted to bundle neurons based on co-activation patterns (storing highly correlated neurons' weights contiguously to increase chunk size beyond 2×). Analysis of co-activation distributions (Figure 12) revealed a power-law pattern where each neuron has a "closest friend" that co-activates 95-100% of the time. However, implementing this bundling strategy "resulted in loading highly active neurons multiple times and the bundling worked against our original intention." The problem: universally active neurons are the "closest friends" of many other neurons, causing redundant loading that outweighs the chunk-size benefit. This negative result is included to "inspire future research on effective neuron bundling."

- **Operating system caching effects (Appendix C):** The paper explicitly benchmarks without OS caching to establish conservative throughput measurements. The authors note that "practical systems may or may not rely on filesystem cache, depending on requirements" and that "these figures can improve if either the inference code or the operating system is allowed to cache some part of the weights." This is a robustness check establishing that the reported speedups are lower bounds — enabling caching could improve performance further, but the paper's measurements do not depend on cache hits for their validity.

- **Nucleus sampling vs. greedy decoding (Figure 8):** Tested for long generation (1000 tokens) on both CPU and GPU. The paper reports: "Nucleus sampling doesn't lead to lower performance in long generation in neither cpu or gpu." The per-token flash latency curves for Nucleus sampling (both CPU and GPU) closely track the greedy decoding curve, confirming that temporal locality of neuron activations is robust to stochastic decoding.

- **Alternative predictor approach for Llama 2 (Section C.5):** Because Llama 2's gate projection with FATReLU provides sparse intermediate outputs, the paper tested using the gate projection itself as the "predictor" (i.e., the sparsity is directly determined by which gate values exceed the FATReLU threshold). "Since gate projects take 1/3 of FFN layer and 5/9 of each transformer block, keeping them in memory will occupy more space in DRAM than having predictors. In fact with window size of 1, this approach resulted in requiring 65% of model size in DRAM." This ablated alternative consumed significantly more DRAM than the trained low-rank predictor approach, validating the predictor design choice for memory-constrained settings.

- **Quantization interaction (Appendix F, Table 6):** The paper briefly examines whether quantization alters activation sparsity patterns. Comparing OPT 6.7B in FP16 vs. a quantized (4-bit) variant across 100 sequences: the percentage of active neurons per layer is nearly identical (Layer 1: 1.56% vs. 1.42%, Layer 16: 2.66% vs. 2.44%, Layer 32: 5.36% vs. 5.45%, Average: 3.30% vs. 3.27%). This validates that "quantization does not alter activation sparsity patterns" and the same predictor and windowing techniques can be applied to quantized models, though actual deployment would require "special 4-bit compute kernels on device, which falls outside the scope of this paper."

- **Speculative decoding integration (Table 3, Section 5.2):** Using λ = 4 draft tokens, speculative decoding on top of the "All" configuration reduces total latency from 84 ms to 60 ms (1.4× speedup). The paper notes this is "close to the original 1.58x speedup of speculative decoding" on the unmodified model, suggesting that the flash-loading system does not fundamentally limit the effectiveness of orthogonal speedup techniques like speculative decoding.

### Critical Assessment

#### Claim 1: "Models can be run up to twice the size of available DRAM"

**Substantially supported, with important boundary conditions.** The paper demonstrates this across five model families (OPT 6.7B, Falcon 7B, Persimmon 8B, Phi-2 2.7B, Llama 2 7B) on three hardware platforms (M1 Max, M2 Ultra, RTX 4090). The DRAM budgets reported — ~52% for OPT, ~53% for Falcon, ~50% for Persimmon, ~65% for Phi-2, ~55% for Llama 2 — all show models running with DRAM allocations between 50-65% of model size, meaning models 1.5-2× larger than available DRAM are indeed running. The power consumption caveat in Section 5.3 (sparse model has lower instantaneous power but higher total energy due to longer generation time) doesn't undermine this claim — it describes an energy trade-off, not a correctness or feasibility limitation. However, the claim implicitly assumes the model has been sparsified (ReLU-activated) and that the remaining FFN sparsity is sufficient to keep the working set within DRAM. For models with lower sparsity (the paper doesn't specify a minimum threshold), the "up to 2×" bound may not hold.

#### Claim 2: "Up to 4× speedup in CPU and 20× speedup in GPU compared to naive loading"

**Supported, but the "up to" framing is doing heavy lifting.** The specific numbers: OPT 6.7B on CPU achieves 4.75× speedup (Table 3), which exceeds the claimed 4×. On GPU (RTX 4090), OPT achieves 26.4× speedup, which exceeds the claimed 20×. However, these are the best-case numbers across all tested configurations. Other models show lower speedups: Persimmon 8B achieves 3.7× on CPU, Phi-2 achieves 2.4×, Llama 2 achieves 3.1×. The 20× figure applies specifically to the highest-throughput GPU (RTX 4090) running OPT 6.7B — not to all GPU configurations (the Metal GPU on M1 Max achieves 4.2× for the same model). The paper is transparent about these per-model, per-platform numbers in Table 3, but the headline claim in the abstract ("up to 4x and 20x increase in inference speed compared to naive loading approaches in CPU and GPU") should be understood as best-case results for the most favorable model-hardware combination, not typical or guaranteed outcomes.

A more subtle issue: the "naive loading" baseline in the I/O latency analysis (Table 2) loads 13.4 GB from flash at 6.10 GB/s, but this baseline assumes the model is loaded in its entirety every forward pass. In practice, even a naive implementation would likely use some form of caching to avoid re-reading embeddings and attention weights. The "Hybrid" baseline (1090 ms I/O) is arguably a fairer comparison — and against Hybrid, the speedup on M1 Max is 12.5× for I/O alone, not 25×. The paper reports both numbers, but the abstract's "4x and 20x" references the total end-to-end speedup against Naive, not the I/O-only speedup against Hybrid.

#### Claim 3: "The method relies on activation sparsity — inference from flash is practical only when the base model's activation pattern is sufficiently sparse"

**Strongly supported, though the precise sparsity threshold is not characterized.** The paper is explicit that this is a sparsity-dependent technique. Table 2 shows that the predictor alone (without windowing) reduces I/O latency from 1090 ms to 738 ms — a modest 1.48× improvement — because throughput collapses from 6.10 GB/s to 1.25 GB/s when reads become sparse. The windowing and bundling techniques are what make this throughput penalty tolerable, not what eliminates it. The paper does not systematically vary sparsity levels to identify a minimum threshold, but the model diversity provides indirect evidence: Phi-2 (lowest sparsity, largest relative accuracy drop) achieves the smallest speedup (2.4× vs. 4.75× for OPT), while OPT (97% sparsity) achieves the largest. Models with lower sparsity would presumably see diminishing returns until the crossover point where selective loading becomes slower than the Hybrid baseline.

The paper does not answer the question: "At what sparsity level does selective loading from flash become *slower* than loading the full FFN sequentially?" The throughput data in Figure 2b and Table 2 provide enough information to estimate this. The Hybrid baseline loads 6.7 GB at 6.10 GB/s (1090 ms). With selective loading at 1.25 GB/s (predictor-only throughput), you could load up to 1.25 GB/s × 1.090 s ≈ 1.36 GB of sparse data before exceeding the Hybrid baseline's latency. If the model's FFN is 8.9 GB (two-thirds of 13.4 GB), the break-even sparsity is 1.36 / 8.9 ≈ 15.3%. If selectivity can load less than ~15% of FFN weights, it wins; if more than ~15%, the Hybrid baseline is faster. With windowing and bundling (2.25 GB/s), the break-even rises to ~27%. These back-of-envelope calculations are not in the paper but would strengthen the sparsity-dependence claim.

#### Missing Experiments and Ablations

- **No profiling of the latency-to-first-byte penalty directly.** Figure 2b shows random read throughput vs. chunk size and thread count, but the paper doesn't report actual "latency to first byte" measurements or break down how much of the per-read time is spent waiting for I/O setup vs. actual data transfer. This makes it harder to assess whether further chunk-size increases (beyond 2× via bundling) would yield diminishing returns or whether the 32-thread parallelism is near-optimal.

- **No end-to-end latency breakdown for the Hybrid baseline in Table 3.** The "Hybrid" rows in Table 3 report only total latency, not the I/O/Mem/Compute breakdown. The I/O latency for Hybrid is reported in Table 2 (1090 ms), but the memory management and compute components are not separately reported. This makes it difficult to assess whether the memory management optimization (Section 3.3) contributes meaningfully beyond the Hybrid baseline, or whether its benefits are only relative to the All configuration.

- **No sensitivity analysis of window size `k` on accuracy.** The paper explores the memory-latency trade-off of varying `k` (Figure 7) but never evaluates whether larger windows affect model outputs. Theoretically, windowing is lossless — it loads all predicted active neurons, just with caching to avoid redundant loads. But the paper doesn't explicitly verify that the cached weight values don't drift (e.g., due to memory corruption or concurrency bugs) over long sequences.

- **No head-to-head comparison with FlexGen in the sub-DRAM regime.** The paper positions itself against FlexGen (Sheng et al., 2023) as a key baseline that "is theoretically bound by the slow throughput of flash to DRAM" (Section 6) in the regime where DRAM < model size. However, no empirical comparison is provided — not even a back-of-envelope calculation showing what FlexGen's latency would be on the same hardware with the same models. This is a missed opportunity to quantify the claimed advantage.

- **Single prompt distribution (C4) for all latency benchmarks.** The C4 dataset is a web-text corpus, but different domains (code, scientific text, multilingual text) might exhibit different activation sparsity patterns and different temporal locality. The paper does not evaluate whether the sparsity and windowing benefits generalize across prompt domains, which matters for real-world deployment where input text varies widely.

- **Limited exploration of multi-batch inference.** The paper explicitly limits scope to "single-batch inference" (Section 8), noting this as a limitation. For server-side deployments where multiple requests are batched together, the selective loading strategy would need to union the active neuron sets across all sequences in the batch. As batch size grows, the union of active neurons approaches the full FFN, potentially eliminating the sparsity benefit entirely. The paper acknowledges this but provides no quantitative analysis of where the break-even batch size lies.

- **Power and energy measurements are qualitative only.** Section 5.3 states that "the sparse model had lower power but higher total energy consumption" but provides no numbers, no measurement methodology, and no breakdown by component (flash I/O vs. DRAM vs. compute). For on-device deployment where battery life is critical, this is a significant omission. The paper acknowledges this as future work but the complete absence of quantitative power data is a weakness for a paper whose primary motivation is "devices with limited memory."

#### Conditional Nature of the Claims

The effectiveness of this approach depends on at least five conditions, all of which hold in the paper's experiments but may not generalize:

1. **Sufficient activation sparsity:** The model must exhibit FFN sparsity above some threshold (roughly >85-90%, based on the break-even analysis above) for selective loading to outperform sequential loading at flash bandwidth. The paper works exclusively with ReLU-activated or FATReLU-activated models that achieve >90% sparsity. Models using GELU, Swish, or other non-sparsifying activations would see dramatically lower benefits — potentially none.

2. **Flash random-read throughput above a device-specific minimum:** The M1 Max achieves ~2.25 GB/s for 32 KiB random reads with 32 threads. Devices with slower flash (e.g., eMMC storage in budget smartphones, older SATA SSDs) would see proportionally higher I/O latency, potentially making selective loading uncompetitive with the Hybrid baseline regardless of sparsity.

3. **Sufficient DRAM for the working set:** The window cache, permanent weights (attention, embeddings, predictors), and KV cache must all fit in available DRAM. If DRAM is so constrained that even the sparse working set doesn't fit (e.g., trying to run a 7B model on a device with 2GB available DRAM), the approach breaks down. The paper's experiments operate at 50-65% of model size, suggesting a practical minimum DRAM of roughly one-third to half the model's total size.

4. **Single-stream (batch size 1) inference:** As noted, batching would reduce the effective sparsity by requiring the union of active neuron sets across sequences. The paper's results are specific to single-stream generation and do not extend to batched inference without modification.

5. **Long-enough sequences to amortize initial cache-filling cost:** The first ~50 tokens have substantially higher latency (Figure 8) while the sliding window cache fills. For very short generations (e.g., single-turn QA with <20 output tokens), the average per-token latency would be significantly higher than the steady-state numbers reported. The paper's benchmarks use 256-token generations, which is long enough to amortize this cost, but the abstract's "4x speedup" claim doesn't account for prompt-length-dependent latency variation.

## 6. Limitations and Trade-offs

### 6.1 The System Requires Activation Sparsity That Many Standard Models Do Not Possess

**The assumption or constraint.** The entire approach — selective weight loading, windowing, and the resulting latency improvements — depends on the existence of extreme activation sparsity in the FFN layers. The paper's OPT 6.7B model exhibits 97% sparsity naturally because OPT uses ReLU. Falcon 7B and Llama 2 7B require explicit fine-tuning (relufication) to replace their native GELU/SwiGLU activations with ReLU/FATReLU to achieve 90-95% sparsity. Phi-2 requires both relufication and distillation to achieve usable sparsity while preserving accuracy. The paper acknowledges this dependency:

> "Our methodology is constructed on the foundation of sparsified networks. Nonetheless, the underlying concept holds potential for broader applications. It can be adapted to selectively load weights in non-sparse networks..." (Section 8)

However, the paper provides no evidence or mechanism for how selective loading would work on non-sparse models. The "potential" extension to non-sparse networks is purely speculative.

**The consequence.** The technique as described does not apply to the vast majority of deployed LLMs. Models using GELU (BERT, GPT-2, GPT-3), SwiGLU (Llama, Llama 2 in their original forms, Mistral, Mixtral), or other smooth activation functions produce dense intermediate activations where nearly all FFN neurons contribute non-zero values. For these models, the low-rank predictor would predict nearly 100% of neurons as active (there is no sparsity to predict), the sliding window would rapidly fill to contain the full FFN, and the system would degenerate to the Hybrid baseline — loading half the model from flash per token at 6.10 GB/s, yielding ~1-2 seconds per token.

Even among models that can be sparsified, the relufication process imposes accuracy costs. Phi-2 drops from 57.0 to 54.3 MMLU after relufication + distillation (Figure 10b). Adding predictors drops it further to 52.0 — a total loss of 5.0 MMLU points. For applications where accuracy matters, this degradation may be unacceptable. The paper does not characterize a minimum sparsity threshold, but the break-even analysis (Section 5, Critical Assessment) suggests that if selectivity loads more than ~15-27% of FFN weights, the approach becomes slower than the Hybrid baseline. This means models must achieve better than ~73-85% sparsity for any speedup, and substantially better than that for the claimed 4-25× gains.

**What evidence exists in the paper.** The sparsity dependency is empirically demonstrated across models. The models with highest sparsity achieve the largest speedups: OPT (97% sparsity, 4.75× on CPU, 26.4× on GPU) vs. Persimmon (lower sparsity, 3.7× on CPU) vs. Phi-2 (lowest sparsity, 2.4× on CPU). The relufication process for Falcon, Phi-2, and Llama 2 is described in detail (Sections 4.1, Appendix C.4, C.5), and the accuracy costs are reported (Table 4, Figure 10b). Table 2 demonstrates what happens without sparsity: the Hybrid baseline achieves 1090 ms I/O latency, which is the theoretical lower bound for dense loading.

**Mitigation status.** The paper does not attempt to solve the sparsity requirement — it treats sparsification as a preprocessing step that is "orthogonal to our proposed method" (Section 4.1). The relufication and distillation procedures are inherited from prior work (Mirzadeh et al., 2023; Song et al., 2024; Liu et al., 2023a) and not contributed by this paper. The suggestion that the approach "can be adapted to selectively load weights in non-sparse networks" (Section 8) is offered as a future direction with no concrete proposal for how selectivity would be determined in the absence of activation sparsity (e.g., importance-based pruning scores, input-dependent gating, or learned retrieval). For practitioners with existing dense models, the paper offers no path to deployment: they would first need to relufy their model (which requires fine-tuning data, compute, and acceptance of potential accuracy degradation), then train per-layer predictors, then reorganize flash storage, before the techniques described here become applicable.

---

### 6.2 The Difficulty Estimation Cost Is Substantial and Not Amortized in the Headline Numbers

**The assumption or constraint.** All components of the system require up-front computation that is not accounted for in the reported per-token latencies. Specifically:

- **Predictor training:** For each layer, 10,000 C4 samples must be processed through the full model to collect (attention output, ground-truth activation mask) pairs, then a low-rank classifier must be trained for 2 epochs. The paper reports "4 hours on an A100 GPU to train each predictor" — per model, not per layer. Since OPT 6.7B has 32 layers, this suggests ~128 GPU-hours total for predictor training, though the exact accounting is ambiguous. For Falcon 7B (36 layers) with larger predictors in later layers, the cost would be higher.

- **Preallocation sizing:** The `Req_i` values that determine DRAM buffer sizes must be calibrated by "measuring maximum cache occupancy on a subset of the C4 validation set" (Section 4.1). This requires running the unmodified model on representative data, with the chosen window size, to observe maximum memory usage.

- **Flash layout reorganization:** The row-column bundling requires physically interleaving up-projection columns and down-projection rows on flash. This is a one-time preprocessing cost that the paper does not quantify but involves reading the full model, restructuring the data, and writing it back to flash.

- **Relufication (for non-ReLU models):** Falcon 7B, Phi-2, and Llama 2 all required fine-tuning to induce sparsity, which the paper quantifies partially (Phi-2 used "a refined-web dataset following Mirzadeh et al. (2023)" with distillation, Figure 10b) but does not report in GPU-hours or wall-clock time.

The paper acknowledges the predictor training and preallocation costs partially but never includes them in any latency or throughput calculation. The reported per-token latencies (Table 3) represent steady-state inference after all preprocessing is complete.

**The consequence.** The headline speedup numbers (4× on CPU, 20× on GPU) apply to amortized deployment where the up-front costs are spread across many inference requests. For one-off or low-volume use cases — generating a few hundred tokens from a model, then moving on — the preprocessing cost could dominate the total time. A practitioner who wants to run an LLM from flash for a single session would spend hours (relufication + predictor training + flash reorganization) before generating their first token, negating any inference-time speedup.

More importantly, the preprocessing pipeline is model-specific. Predictors trained for OPT 6.7B cannot be reused for Falcon 7B or Persimmon 8B — they are trained per-layer on that model's specific activation patterns. Flash layout reorganization is similarly model-specific. This means the approach does not enable "plug and play" inference from flash on arbitrary models; each new model or model variant requires its own full preprocessing pipeline. For practitioners who want to experiment with multiple models, the preprocessing cost multiplies.

**What evidence exists in the paper.** The predictor training cost is stated explicitly: "10000 samples from the C4 training dataset to do the training for 2 epochs. It took 4 hours on an A100 GPU to train each predictor" (Section 3.1). However, the ambiguous phrasing — "to train each predictor" — could mean 4 hours per layer (implying 128+ GPU-hours for a 32-layer model) or 4 hours total for all predictors. The relufication cost is not quantified beyond noting that distillation "improves the results" and that the process used a "refined-web dataset" (Appendix C.4). Flash reorganization cost is not mentioned. None of these costs appear in any latency table or budget calculation.

**Mitigation status.** The paper does not address preprocessing cost amortization at all. There is no analysis of how many inference tokens must be generated before the up-front cost is recovered, no suggestion for sharing predictors across models or layers, and no exploration of whether predictors could be trained more cheaply (e.g., with fewer samples, fewer epochs, or online adaptation during inference). The paper frames these as one-time costs without examining their magnitude relative to the per-token savings.

---

### 6.3 Single-Batch Inference Constraint and the Batching Collapse Problem

**The assumption or constraint.** All experiments in the paper operate on **single sequences** — batch size 1. This is stated explicitly but almost in passing:

> "In our experiments, we process sequences individually, running only one sequence at a time" (Section 4.1)

and acknowledged as a limitation:

> "Currently, our study is limited to single-batch inference... expanding this to include more complex scenarios like prompt processing and multi-batch inference are valuable areas for further investigation" (Section 8)

The assumption is baked into the entire system design. The sliding window cache stores neurons active for recent tokens from a single sequence. The memory manager's preallocation is sized for single-sequence cache occupancy. The predictor processes one attention output at a time. There is no mechanism for handling multiple concurrent sequences that would require unioning active neuron sets.

**The consequence.** When multiple sequences are batched together (standard practice for server-side LLM serving to maximize throughput), each token in the batch requires its own set of active neurons. The system must load the **union** of all active neurons across all sequences to compute the FFN layer correctly — because the matrix multiplication operates on the batched activation tensor, and if a neuron is active for any sequence in the batch, its corresponding up-projection column and down-projection row are needed for the computation.

As batch size `B` increases, the probability that any given neuron is active for at least one sequence in the batch grows rapidly. Under reasonable independence assumptions, if a neuron has activation probability `p` for a single sequence, the probability it fires for at least one of `B` sequences is `1 - (1 - p)^B`. With `p = 0.03` (97% sparsity) and `B = 8`, the union probability is `1 - 0.97^8 ≈ 0.22` — a 7.3× increase in expected neurons loaded per batch. With `B = 32`, the union probability reaches `1 - 0.97^32 ≈ 0.62` — nearly two-thirds of all FFN neurons. The sliding window compounds this: the union must be maintained across temporal windows for all sequences in the batch. At modest batch sizes (8-16), the system would likely degenerate to loading the majority of FFN weights per token, making it slower than the Hybrid baseline.

This means the technique is fundamentally incompatible with throughput-oriented batched inference. It is suitable only for latency-sensitive, single-stream scenarios — such as on-device interactive chat, where one user generates one sequence at a time. For any server-side deployment that batches requests from multiple users, the approach offers no benefit and may be actively harmful compared to loading the full FFN sequentially.

**What evidence exists in the paper.** None. The paper does not experiment with batch sizes greater than 1, does not measure how the union of active neuron sets grows with batch size, and does not provide even a theoretical analysis of the batching collapse problem. The acknowledgment in Section 8 is purely forward-looking. The negative result on co-activation bundling (Appendix D) provides indirect evidence that union effects are problematic: highly active neurons become "closest friends" of many others, causing redundant loading — but this is within a single sequence, not across batched sequences.

**Mitigation status.** Not addressed. The paper mentions multi-batch inference as "valuable areas for further investigation" (Section 8) but offers no proposals for how to handle the union problem — for instance, through batch-aware predictor thresholds, sequence-level rather than batch-level FFN computation, or adaptive batching that groups sequences with similar activation patterns. The limitation fundamentally constrains the deployment scenarios where the technique applies.

---

### 6.4 The Approach Has Not Been Demonstrated on Non-Sparsified, Widely-Deployed Models

**The assumption or constraint.** Every model evaluated in the paper either uses ReLU natively (OPT 6.7B, Persimmon 8B) or has been explicitly fine-tuned to use a sparsity-inducing activation function (Falcon 7B with ReLU, Llama 2 7B with FATReLU, Phi-2 with ReLU relufication). The paper's evaluation does not include any model with GELU, SwiGLU, or other smooth activation functions in their original form — which describes the overwhelming majority of currently deployed and widely-used open-source LLMs (Llama 2, Llama 3, Mistral, Mixtral, Gemma, Qwen, DeepSeek, etc.).

The paper acknowledges this indirectly through its reliance on sparsified variants and through the limitation statement:

> "Our methodology is constructed on the foundation of sparsified networks" (Section 8)

but does not frame this as an evaluation gap. The claim that techniques are "mostly independent of architecture" (Section 4.1) refers to transformer architectural variants, not to activation function choice, which is the critical dependency.

**The consequence.** A practitioner using a standard Llama 2 7B model (SwiGLU activation) cannot apply the techniques described in this paper without first relufying the model — a process that the paper itself shows degrades accuracy (Llama 2 7B goes from 41.8 to 38.96 MMLU after FATReLU sparsification; Section C.5) and requires access to fine-tuning data and compute that may not be available. The paper provides no evidence that the predictor-and-windowing approach would produce any speedup on a dense model. The "predictor" on a dense model would simply learn to predict "all neurons active" (since nearly all are), eliminating any data transfer reduction.

More subtly, the paper does not evaluate whether the relufication + predictor pipeline preserves model quality on tasks beyond the reported zero-shot benchmarks. MMLU, ARC, and HellaSwag test factual knowledge and shallow reasoning; they do not test instruction following, multi-turn consistency, coding ability, or safety properties. A practitioner considering relufication to enable flash inference would need to know whether these capabilities are preserved, but the paper provides no such evaluation.

**What evidence exists in the paper.** The evaluation is comprehensive within its scope, covering five model families across multiple sizes (2.7B to 8B). But the scope itself is restricted to sparsified or natively-sparse models. Table 4 provides zero-shot accuracy comparisons between original, relufied, and predictor-equipped models, showing that accuracy is largely preserved (or degraded acceptably). However, no dense-model ablation is attempted — for any model, the "without sparsification" variant is never run through the full flash-loading system to measure what latency would result. The paper could have tested this trivially: run OPT 6.7B with the predictor always returning "all neurons active" (simulating a dense model) and measure the resulting latency. This would have provided a direct empirical bound on the sparsity requirement. Its absence is a notable omission.

**Mitigation status.** The paper does not claim applicability to dense models — it is transparent that sparsification is a prerequisite. The limitation statement in Section 8 acknowledges this implicitly but frames it optimistically: "the underlying concept holds potential for broader applications... to selectively load weights in non-sparse networks... contingent on the specific requirements of the input prompt." This is hand-waving; no concrete mechanism for selectivity without sparsity is proposed. For practitioners, the limitation is binary: if you have a dense model, this paper does not help you; if you are willing to sparsify, the paper shows you can then achieve speedups, but you bear the accuracy risk and engineering cost of sparsification yourself.

---

### 6.5 Power and Energy Consumption Are Qualitatively Worse Than Dense Inference, Without Quantification

**The assumption or constraint.** The paper reports latency improvements but explicitly notes that the energy picture is less favorable:

> "In evaluating the efficiency of our method, we compared the power consumption of our sparse model approach with that of generating tokens using a dense model of similar size. While the power usage (energy per unit of time) of the sparse model was lower than that of the dense model, the extended duration required for token generation resulted in the sparse model having a higher total energy consumption" (Section 5.3)

The paper provides no numbers — no watts, no joules per token, no measurement methodology, no breakdown by component (flash reads vs. DRAM vs. compute vs. idle). The entire power analysis consists of this single paragraph. The limitation is explicitly deferred:

> "A systematic and quantitative evaluation of the exact power usage pattern is left as a future work" (Section 5.3)

**The consequence.** For on-device deployment — the paper's primary motivating use case — battery life is often as important as latency, sometimes more so. A technique that achieves 4× lower latency but consumes (say) 2× more energy per token may be unacceptable for battery-constrained devices like smartphones, where every joule counts. The paper's qualitative statement that the sparse model has "higher total energy consumption" is directionally concerning: it suggests the approach trades energy efficiency for latency, which is the opposite of what mobile deployments typically need (where background efficiency matters more than peak speed).

The mechanism for higher energy consumption is plausible: flash reads consume significantly more energy per byte than DRAM reads (NAND flash requires charge pumps and error correction; DRAM reads are capacitive). The sparse model performs many small random flash reads (even with bundling, the reads are random-access at the flash controller level) rather than fewer large sequential reads. Each random read wakes the flash controller, potentially spins up power-hungry components, and incurs per-operation energy overhead that dominates the per-byte transfer energy for small reads. The total energy could easily be higher even if instantaneous power (watts) is lower because the generation takes longer.

Without quantification, a practitioner cannot make an informed trade-off between the latency gains and the energy penalty. A smartphone manufacturer considering this technique would need to know: "At 4× speedup, how many minutes of battery life do I lose per 1000 tokens generated compared to cloud offloading? Compared to a smaller model that fits entirely in DRAM?" The paper provides no basis for answering these questions.

**What evidence exists in the paper.** None — by the authors' own admission. The power consumption discussion (Section 5.3) is the only reference to energy, and it contains zero measurements. The paper does not report a single joule or watt figure anywhere. This is a significant gap for a paper whose abstract and introduction emphasize "personal devices" and "devices with limited DRAM capacity" — contexts where energy is a first-class constraint.

**Mitigation status.** Not addressed. The paper explicitly defers power analysis to future work (Section 5.3, Section 8). This is not a hidden limitation — the authors are transparent about it — but the complete absence of even a preliminary measurement (e.g., wall-plug power during a generation run on the MacBook) weakens the practical deployment story. A practitioner reading this paper learns that the technique is faster but may be less energy-efficient, with no way to assess whether the energy penalty is modest (5-10%) or prohibitive (2-5×).

---

### 6.6 Flash Wear and Device Longevity Are Not Considered

**The assumption or constraint.** The paper's entire approach is built on continuous, per-token reads from flash memory. For a 7B model with windowing, each generated token requires reading roughly 0.2 GB from flash (Table 2, OPT 6.7B). Generating 256 tokens thus reads approximately 51 GB from flash per sequence. An interactive chat session generating thousands of tokens would read hundreds of gigabytes. Over months or years of use, the total flash read volume could reach into the hundreds of terabytes or more.

NAND flash memory has a finite lifespan measured in **program/erase (P/E) cycles** — typically 300-3000 for consumer TLC/QLC NAND, or 10,000-100,000 for enterprise MLC NAND. Reads do not consume P/E cycles directly, but they do cause **read disturb** — a phenomenon where reading one page in a NAND block can gradually alter the charge levels in adjacent pages, potentially causing data corruption if not managed by the flash controller through background data refreshing (read reclaim). Read disturb thresholds are typically in the hundreds of thousands to millions of reads per block.

The paper does not mention flash endurance, read disturb, or device longevity anywhere. The assumption — implicit and unstated — is that flash storage can sustain the read volume of LLM inference indefinitely without reliability degradation.

**The consequence.** On consumer devices with QLC NAND (common in modern laptops and smartphones due to cost/density advantages), read disturb thresholds are lower than on enterprise SSDs. Continuous LLM inference — which might run for hours in an interactive setting — could generate read volumes that approach or exceed read disturb limits over the device's lifetime. The flash controller mitigates this through background data refreshing, but this consumes additional write cycles (which do count against P/E endurance) and I/O bandwidth (invisible to the application but competing for flash controller resources).

More practically, the flash read workload generated by LLM inference competes with all other I/O on the device — application launches, file saves, media playback, OS paging. A device running this system as a background process (say, an always-on assistant) would subject its flash storage to a continuous, high-volume read workload that it was never designed for. The paper does not address whether this could cause premature device failure, trigger thermal throttling under sustained load, or degrade the user experience for other applications competing for flash bandwidth.

**What evidence exists in the paper.** None. The words "endurance," "wear," "lifespan," "read disturb," "P/E cycle," and "longevity" appear nowhere in the paper. The long-generation test (Figure 8, 1000 tokens) demonstrates thermal stability over ~100 seconds of continuous generation but does not address multi-hour or multi-day sustained operation. The paper's focus is entirely on per-token latency, not on system-level reliability or device health over time.

**Mitigation status.** Not addressed. This is understandable — the paper is a first demonstration of feasibility, and device longevity is typically a second-order concern for research prototypes. However, for a technique explicitly targeted at "personal devices" (Section 1, Section 4.1, Section 7) where users expect their hardware to function reliably for years, the absence of any endurance analysis is a practical limitation. A deployer would need to answer: "If a user runs this for 2 hours per day generating text, will their SSD fail prematurely?" The paper provides no basis for answering this question.

## 7. Implications and Future Directions
- What changes now
  - Treating SSD as the primary store reshapes how on-device LLM inference can be architected. With the right I/O patterns and minimal DRAM residency, devices can serve models roughly twice their DRAM size with acceptable latency (Figure 1; Abstract; Table 3).

- Practical applications
  - Private, offline assistants on laptops; edge deployments where cloud is unavailable; developer workflows where larger models can run locally without GPU-class DRAM.
  - Appendix F sketches smartphone feasibility when combined with 4‑bit quantization, provided device kernels support low‑bit compute and the same sparsity holds (Table 6).

- Follow-up research enabled
  - Smarter bundling: The co-activation negative result (Appendix D; Figure 12) suggests investigating more sophisticated bundling strategies (e.g., disjoint cluster bundles that avoid reloading hot neurons).
  - Multi-batch, multi-session memory managers: Extending the sliding window to shared/evolving caches across conversations or users.
  - Joint design with compression: Integrating quantization/pruning with flash-aware layouts and predictors while preserving accuracy.
  - Power/thermal modeling: A systematic measurement of energy vs. latency trade-offs and thermal constraints for long sessions (Section 8).
  - Architectures with built-in flash-friendly layouts: File formats or weight sharding schemes aligned to predictor access patterns and SSD block sizes.
  - Broader decoding strategies: The speculative decoding prototype (Table 5; Section 5.2) could be combined with other parallel or early-exit decoding methods.

Quoted highlights (grounding in the paper)
- Flash vs DRAM capacity/bandwidth:
  > “Flash memory offers significantly higher capacity but suffers from much lower bandwidth compared to DRAM…” (Figure 2a)
- Random-read throughput improves with chunk size and threads:
  > “The throughput for random reads in flash memory increases with the size of sequential chunks and the number of threads.” (Figure 2b)
- End-to-end improvements with the full method:
  > For `OPT‑6.7B` on NVIDIA GPU: “Naive 2218 ms … All 84 ms” (Table 3)
- Stepwise I/O reduction:
  > “Using predictors, windowing, and bundling… I/O latency [drops] from 2196 ms to 87 ms” for `OPT‑6.7B` on M1 Max (Table 2)
- Capability claim:
  > “Run models 2x larger than the device’s DRAM capacity and speed up inference up to 4x, 7x, and 20x compared to naive implementations in CPU, Metal, and NVIDIA GPU backends, respectively.” (Section 1; also reflected in Table 3)

In sum, LLM in a flash is a well-motivated, hardware-aligned design that demonstrates sizeable latency gains by pairing activation-aware selective loading with flash-friendly I/O and DRAM-efficient data structures. It opens a clear path for larger on-device models and invites a wave of systems-and-algorithms co-design around SSD-centric inference.
