# QERL: BEYOND EFFICIENCY – QUANTIZATION ENHANCED REINFORCEMENT LEARNING FOR LLMS

**ArXiv:** [2510.11696](https://arxiv.org/abs/2510.11696)

## 🎯 Pitch

QeRL introduces a transformative reinforcement learning framework for LLMs by synergizing ultra-efficient 4-bit NVFP4 quantization, low-rank adaptation (LoRA), and an adaptive quantization noise (AQN) scheduler. This trio not only slashes memory use and rollout times—enabling end-to-end RL speedups of 1.2–1.5× (and up to 2× in generation)—but also boosts exploration and final accuracy, achieving results on par with full-parameter fine-tuning on demanding reasoning tasks. The innovation makes scalable RL training feasible even for massive models (up to 32B) on a single H100 GPU, unlocking faster, cheaper, and more capable LLMs for real-world deployments.

---

## 1. Executive Summary

This paper introduces **QeRL**, a **Quantization-enhanced Reinforcement Learning** framework that combines NVFP4 precision quantization with Low-Rank Adaptation (LoRA) to accelerate the rollout phase of RL training for LLMs while reducing memory overhead. The core insight is that quantization noise, rather than degrading training as it does in supervised fine-tuning, increases policy entropy and enhances exploration—functionally analogous to parameter-space noise injection in traditional RL—enabling quantized models to discover better reasoning strategies. To transform static quantization noise into a dynamically controlled exploration mechanism, QeRL introduces an **Adaptive Quantization Noise (AQN)** mechanism that injects channel-wise Gaussian noise modulated by an exponential decay schedule. Experiments on the GSM8K and BigMath mathematical reasoning benchmarks using Qwen2.5-Instruct models (3B to 32B) demonstrate that QeRL delivers over 1.5× speedup in the rollout phase and up to 1.8× end-to-end training speedup over QLoRA, while achieving 90.8% on GSM8K and 77.4% on MATH 500 for the 7B model—surpassing both 16-bit LoRA and QLoRA and matching full-parameter fine-tuning accuracy, establishing that quantization can serve as a performance-enhancing exploration mechanism in RL training rather than merely a compression technique.

## 2. Context and Motivation

### The Core Problem: RL Training for LLMs Is Prohibitively Expensive

The paper tackles a practical bottleneck that stands between current LLM training practice and the emerging consensus that reinforcement learning is essential for reasoning capabilities. Over the past two years, the field has converged on a critical finding: supervised fine-tuning (SFT) alone is insufficient for developing robust multi-step reasoning in LLMs. SFT trains models to imitate explicit reasoning traces—essentially memorizing solution patterns—but this approach risks producing models that replicate surface-level structures rather than engaging in genuine logical deduction. As the authors note in Section 1:

> "this approach risks promoting imitation rather than encouraging genuine reasoning"

In contrast, reinforcement learning uses verifiable reward signals (e.g., checking whether a math answer is numerically correct) to support adaptive learning. RL allows the model to explore diverse reasoning paths—including ones that make mistakes, backtrack, or take circuitous routes—and receive credit only when the final answer is correct. This exploration-rich training paradigm has driven remarkable results: DeepSeek-R1 (DeepSeek-AI, 2025) demonstrated that RL alone can produce strong reasoning capabilities competitive with proprietary models, and the broader literature has shown that models trained with RL generalize better to novel problem structures than those trained with SFT alone (Chu et al., 2025).

However, RL training is enormously resource-intensive. The paper identifies four distinct sources of computational burden:

1. **Multi-model memory pressure.** Algorithms like GRPO (Group Relative Policy Optimization) require the simultaneous loading of multiple models—at minimum a policy model and a reference model—into GPU memory. For large reasoning-focused models (e.g., 32B, 70B parameters), this places memory demands far beyond what a single high-end GPU can provide.

2. **Expensive rollout generation.** The rollout phase, where the policy model generates complete reasoning traces for a batch of prompts, is particularly costly. Each rollout involves autoregressive token generation over long sequences—the paper's training setup uses completion lengths of up to 8192 tokens for BigMath problems. For complex reasoning tasks, the model may generate thousands of tokens per sample, and RL requires many such samples (8 per prompt for GSM8K, 16 for BigMath) to compute robust advantage estimates.

3. **Multistage training pipeline.** Each RL training step involves rollouts, reward computation (running the verifier), log-probability evaluation on both the current and old policies, advantage estimation, and gradient updates. These stages create a complex pipeline where computational bottlenecks compound.

4. **Sample inefficiency.** RL inherently requires many interactions to learn effective policies. The model must explore a vast space of possible reasoning traces, most of which are incorrect, to discover the minority that lead to correct answers.

The practical consequence is scale-limiting: training a 32B model with RL on a single H100 80GB GPU was, prior to this work, not feasible. Organizations wanting to apply RL to reasoning models faced either multi-GPU distributed training (with associated engineering complexity) or were restricted to smaller models.

### Why This Problem Matters: The SFT-to-RL Transition

The importance of this problem extends beyond academic interest into the practical economics of LLM development. The field is undergoing a structural shift: the dominant post-training paradigm is moving from SFT-heavy to RL-heavy pipelines. Evidence for this shift includes the release of DeepSeek-R1 (trained primarily with RL), the development of specialized RL frameworks for language models (DAPO, GSPO), and the broader recognition that RL-trained models exhibit properties—self-verification, error recovery, exploration of multiple solution strategies—that SFT models lack.

In this new paradigm, the **efficiency of RL training** becomes a first-order concern. The paper's framing implicitly challenges a status quo assumption: that RL's computational cost is an unavoidable price of improved reasoning. By developing a quantization-based framework that simultaneously reduces memory and accelerates computation while improving exploration, QeRL directly addresses the economic viability of RL training at scale.

The paper also speaks to a specific hardware constraint that is relevant across the industry. While model sizes continue to grow (from 7B to 70B to 405B and beyond), the memory capacity of individual GPUs grows more slowly. Nvidia's H100 provides 80GB—comfortable for inference with 7B models, tight for 32B models with multiple copies loaded for RL, and impossible for 70B+ models without extensive sharding. Quantization at the 4-bit level (reducing model memory to ~25-30% of BF16) fundamentally changes this equation: models that previously required multiple GPUs can now fit on a single GPU, democratizing access to RL training for smaller research labs and enabling faster experimental iteration.

### Where Existing Approaches Fall Short

The paper identifies three categories of prior approaches to improving RL efficiency and explains why each is inadequate.

**Parameter-Efficient Fine-Tuning (PEFT) Alone Doesn't Address the Rollout Bottleneck**

The most natural approach to reducing RL's memory cost is to reduce the number of trainable parameters using methods like LoRA (Hu et al., 2022). By freezing the base model weights and training only low-rank adapter matrices, LoRA dramatically reduces the optimizer state memory (no need to store Adam moments for billions of frozen parameters) and the gradient memory (gradients only flow through the adapter layers).

Tina (Wang et al., 2025) demonstrated that this PEFT approach can work for RL training of reasoning models—small models can gain reasoning capabilities through LoRA-based RL. However, the paper identifies a critical limitation:

> "similar to LoRA in SFT, these methods fail to address the core issue of slow rollout speeds"

LoRA reduces the trainable parameter burden but does nothing for the forward-pass speed during rollout generation. The base model weights—still in 16-bit precision—must be loaded and computed for every token of every rollout. The rollout phase (generating reasoning traces) often dominates the total training time because it involves autoregressive generation of long sequences, and LoRA provides no acceleration here. The memory savings from LoRA are genuine but incomplete: the 16-bit base model weights still occupy substantial GPU memory during rollout, limiting the maximum model size that can be trained.

**QLoRA's 4-bit Quantization Addresses Memory but Degrades Speed**

QLoRA (Dettmers et al., 2023a) extended LoRA by quantizing the frozen base model to 4-bit NormalFloat (NF4) precision. This addresses the memory limitation: the base model weights occupy only ~25-30% of their BF16 footprint, making larger models feasible on a single GPU. For SFT, QLoRA has been widely successful and is now a standard tool.

However, the paper highlights a performance property of NF4 that makes it unsuitable for RL's rollout-intensive workloads:

> "using QLoRA in RL slows rollouts by 1.5–2×, further reducing efficiency. This slowdown occurs because QLoRA relies on NormalFloat 4-bit (NF4) precision, which requires unpacking and mapping to floating-point values via a lookup table before matrix multiplication."

This is a subtle but critical architectural detail. NF4 is a data-dependent quantization format optimized for normally distributed weights—the quantization levels are not uniformly spaced but instead placed at quantiles of the Gaussian distribution. To perform matrix multiplication with NF4 weights, the hardware must (1) look up each 4-bit index in a table to recover the corresponding floating-point value, (2) assemble these values into a floating-point matrix, and then (3) perform standard floating-point multiplication. This unpacking step adds overhead that makes NF4 inference *slower* than BF16 inference, despite the reduced memory footprint. For RL—where rollouts involve generating thousands of tokens per step—this slowdown is devastating: QLoRA might reduce memory but increase total training time, defeating the purpose of efficiency optimization.

**FlashRL's Quantized Rollouts Create a Precision Mismatch Problem**

FlashRL (Liu et al., 2025a) takes a different approach: use an 8-bit quantized model for the fast rollout phase but keep a separate 16-bit model for computing the log-probabilities needed for policy gradient updates. This is motivated by the observation that rollouts (which only require forward-pass sampling) can tolerate lower precision than log-probability evaluation (which feeds into gradient computation and requires numerical stability).

The paper identifies two problems with this approach:

> "precision mismatches between the rollout model and logits model (e.g., 8-bit vs. 16-bit) require importance sampling to correct discrepancies, necessitating both 8-bit and 16-bit models to run simultaneously, which increases memory usage."

First, the **importance sampling correction** itself adds computational overhead. When the rollout model's output distribution differs from the logits model's distribution (due to quantization-induced probability shifts), the policy gradient estimator must reweight samples using importance weights—the ratio of probabilities under the two models. Computing these weights requires evaluating both models on each generated token, which partially negates the speed advantage of the quantized rollouts.

Second, and more critically for the paper's GPU-constraint argument, FlashRL **requires both models in memory simultaneously**. The memory savings from quantizing the rollout model are offset by the need to retain the 16-bit model for log-probability computation. This makes FlashRL unsuitable for the memory-constrained single-GPU training scenario that QeRL targets.

### The Quantization Noise Paradox: A Problem Becomes a Feature

The paper's central conceptual move is to reframe quantization noise—universally considered a source of degradation in SFT and inference—as a potential benefit in RL. This reframing is what distinguishes QeRL from straightforward engineering optimizations.

**The conventional wisdom:** In supervised fine-tuning, quantization is understood as a necessary evil—a compression technique that trades precision for efficiency. The noise introduced by mapping continuous weights to discrete quantization levels corrupts the model's output distribution, causing slight degradations in accuracy and calibration. QLoRA's contribution was to show that this degradation can be made acceptably small by using data-dependent quantization formats (NF4) and by training adapter layers that compensate for quantization errors. But the fundamental view remained: quantization noise is harmful, and the goal is to minimize its impact.

**The paper's counterintuitive observation:** When applying PEFT-based RL to quantized models, the authors observe that 4-bit quantized models *consistently outperform* their 16-bit counterparts. Specifically:

- Reward curves show faster convergence (steeper upward trend, earlier plateau).
- Final evaluation accuracy is higher.
- This advantage holds across multiple quantization formats (NVFP4, MXFP4, NF4) but is strongest for NVFP4 and MXFP4.

This finding contradicts the SFT paradigm where quantization degrades performance. The paper provides a mechanistic explanation illustrated in Figure 3: quantization noise increases the policy's sampling entropy. When a quantized model produces a probability distribution over the vocabulary, the small systematic errors introduced by weight quantization propagate through the network layers (as described in Equation 5: $\Delta\epsilon = Q(\theta) - \theta$), perturbing the final logits before softmax normalization. This perturbation "flattens" the output distribution—reducing the probability mass on the single most likely token and spreading it across plausible alternatives—resulting in higher entropy $H(\pi(\cdot|q)) = -\sum_{o_t \in V} \pi(o_t|q) \log \pi(o_t|q)$.

In the context of RL, this entropy increase is not a bug but a feature. RL thrives on exploration: the model must sample diverse reasoning paths to discover which ones lead to correct answers, and a model that is overconfident in its initial (potentially incorrect) strategy will fail to find better alternatives. Quantization noise provides a form of **implicit parameter-space exploration**—analogous to the deliberate noise injection techniques developed in the traditional RL literature (e.g., parameter noise for continuous control in Plappert et al., 2017; noisy networks in Fortunato et al., 2018). The crucial difference is that in QeRL, this exploration mechanism emerges "for free" as a byproduct of quantization, rather than requiring explicit noise injection infrastructure.

**The limitation of static quantization noise:** However, the paper identifies that this implicit exploration mechanism has a critical shortcoming. Unlike deliberately injected noise in traditional RL—which can be sampled randomly and independently at each training step, and whose magnitude can be adjusted via a schedule—quantization noise is **static and deterministic**. Once the weights are quantized, the noise pattern $\Delta\epsilon = \hat{W} - W$ is fixed. It provides a constant exploration bonus throughout training, regardless of whether the model is in an early phase (where high exploration is beneficial) or a late phase (where exploitation of learned strategies should dominate).

This static nature fails to respect the fundamental **exploration-exploitation trade-off** that governs effective RL training. Early in training, the model knows little about which reasoning strategies work; high exploration helps it sample broadly and discover promising approaches. Late in training, the model has identified effective strategies; excessive exploration introduces variance that prevents convergence to optimal behavior.

### How QeRL Positions Itself Relative to Prior Work

The paper positions QeRL as addressing the limitations of all three prior efficiency approaches simultaneously while adding a novel exploration-enhancement capability:

**Against LoRA (PEFT-only):** QeRL goes beyond LoRA by quantizing the base model weights to NVFP4, achieving both memory reduction (model size drops to ~25-30% of BF16) and rollout acceleration (via hardware-optimized Marlin kernels for NVFP4×BF16 matrix multiplication). The rollout speed improvement is the key distinction—LoRA leaves the forward pass untouched, while QeRL fundamentally accelerates it.

**Against QLoRA (NF4 + LoRA):** QeRL replaces NF4 with NVFP4, solving the unpacking overhead that makes QLoRA *slower* than BF16. NVFP4 uses a dual-scaling floating-point format (E2M1 mantissa/exponent with FP8 block-wise scalers) that is natively supported by Marlin kernels on H100 GPUs. This enables direct hardware-accelerated matrix multiplication without the lookup table indirection that NF4 requires. The result is that QeRL is faster than BF16 LoRA (1.2-1.5×), whereas QLoRA is slower (0.7-0.8×).

**Against FlashRL (dual-precision rollouts):** QeRL avoids the precision mismatch problem entirely by using the *same quantized model* for both rollouts and log-probability evaluation. The LoRA adapters are trained in BF16 precision and merged with the quantized base weights, producing consistent output distributions for both sampling (rollouts) and scoring (log-probability computation). This eliminates the need for importance sampling corrections and the memory cost of maintaining dual 8-bit/16-bit models simultaneously—QeRL keeps only the quantized base model and the small LoRA adapter matrices in memory.

**Beyond efficiency—exploration as a first-class benefit:** The most distinctive aspect of QeRL's positioning is its claim that quantization is not merely a tolerated compromise but an **active contributor to training quality**. The Adaptive Quantization Noise (AQN) mechanism explicitly builds on the observation that quantization noise increases entropy, transforming a static byproduct into a dynamically controlled exploration tool. This reframes the entire quantization-for-training conversation: instead of asking "how can we minimize quantization's damage?", QeRL asks "how can we harness quantization's noise to improve RL training?"

This positioning is reinforced by the paper's architectural decisions. QeRL does not treat quantization and exploration as separate concerns—it integrates them through the Noise Merging technique (Equation 9-10, Figure 6), which folds the injected noise vector into the LayerNorm parameters to avoid additional memory overhead or kernel compatibility issues. The result is a framework where efficiency (quantization) and effectiveness (exploration) are mutually reinforcing rather than in tension.

## 3. Technical Approach

### 3.1 Reader Orientation

QeRL is a training framework that wraps standard LLM reinforcement learning algorithms (GRPO, DAPO) with 4-bit weight quantization and a controlled noise-injection mechanism to simultaneously accelerate the expensive rollout generation phase and improve the model's ability to explore diverse reasoning strategies. The problem it solves is that RL training for LLMs requires running multiple large models in GPU memory while generating thousands of long reasoning traces—a combination that makes training slow and memory-hungry—and the solution is to quantize the frozen base model to NVFP4 (which reduces memory and accelerates matrix multiplication via hardware-optimized kernels), adapt only small LoRA matrices during training, and inject an exponentially decaying Gaussian noise schedule into the LayerNorm parameters to transform static quantization error into a dynamic exploration mechanism that boosts reward convergence.

### 3.2 Big-Picture Architecture (Diagram in Words)

The QeRL system has five major components that interact during each RL training step:

1. **NVFP4-Quantized Base Model ($\tilde{\pi}_\theta$​)** — the frozen pretrained LLM whose weights are compressed to 4-bit NVFP4 floating-point format using activation-aware weight quantization (AWQ). This model serves as the backbone for all forward-pass computations (rollout generation and log-probability evaluation) but receives no gradient updates. Its 4-bit weights are stored persistently and loaded once.

2. **LoRA Adapter Matrices ($\theta_{\text{lora}}$​, rank $r$)** — small, trainable low-rank matrices ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$) attached to the frozen quantized weight matrices in the attention projections ($W_q, W_k, W_v, W_o$) and feed-forward projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). Only these matrices (and the LayerNorm parameters) receive gradient updates during RL training. The effective policy model is $\pi_\theta = \tilde{\pi}_\theta + \theta_{\text{lora}}$.

3. **Adaptive Quantization Noise (AQN) Module** — a noise-injection mechanism that samples independent Gaussian noise vectors $Z_{\text{noisy}} \sim \mathcal{N}(0, \sigma^2 I)$ for each forward pass and merges them into the RMSNorm scaling parameters (Equation 10) before the quantized linear layers. The noise standard deviation $\sigma$ follows an exponential decay schedule (Equation 8) across $K$ evenly-spaced stages, transitioning from $\sigma_{\text{start}} = 1 \times 10^{-2}$ to $\sigma_{\text{end}} = 5 \times 10^{-4}$. This transforms the static deterministic quantization error into a dynamic, controllable exploration signal.

4. **Marlin-Accelerated Rollout Engine** — a specialized inference kernel (Frantar et al., 2024) that performs NVFP4 × BF16 matrix multiplication directly on H100 GPU tensor cores without unpacking. This kernel handles the dual-scaling NVFP4 format: a per-tensor FP32 global scaler $S_{\text{FP32}}$ and a fine-grained tensor of block-wise FP8 (E4M3) scalers $S_{\text{E4M3}}$ applied to blocks of 16 elements. Rollouts use this kernel for fast autoregressive token generation.

5. **RL Policy Optimizer (GRPO or DAPO)** — the outer-loop RL algorithm that samples $G$ candidate outputs per prompt (8 for GSM8K, 16 for BigMath), scores them with a rule-based reward function (exact-answer matching), computes group-relative advantages $A_i$ (Equation 4), and updates the LoRA adapter parameters via policy gradient ascent with clipping (Equation 3).

**Information flow per training step:** A batch of prompts $D_b$ is sampled from the training dataset → The policy model $\pi_\theta$ (NVFP4 base + LoRA + AQN noise at current $\sigma$) generates $G$ complete reasoning traces per prompt using the Marlin-accelerated rollout engine, producing candidate outputs $\{o_i\}_{i=1}^G$ → A rule-based reward function $r_\phi$ scores each output (1 if the extracted answer matches the ground truth, 0 otherwise) → Group-relative advantages $\hat{A}_{i,t}$ are computed per token by normalizing rewards within each prompt group (Equation 4) → The policy gradient objective (Equation 3) is maximized over $\mu$ inner-loop updates, updating only the LoRA matrices and RMSNorm parameters → The AQN noise level $\sigma$ is decayed to the next stage value every $\lfloor M/K \rfloor$ steps → The process repeats for $I$ total iterations.

### 3.3 Roadmap for the Deep Dive

- **First**, the **NVFP4 quantization format and dequantization mechanism** (Equations 1, 6), because all forward-pass computations—rollout generation, log-probability evaluation, and reward scoring—depend on the quantized weights, and the dual-scaling structure of NVFP4 is what enables both memory reduction and Marlin kernel acceleration.

- **Second**, the **LoRA adaptation within quantized layers** (Equation 2), which defines how the trainable adapter matrices interact with the frozen quantized base weights to produce the effective policy model, and explains the parameter-efficiency-memory tradeoff.

- **Third**, the **quantization noise as exploration mechanism** analysis (Equation 5, Figure 3, Figure 5), which establishes the paper's central empirical finding—that quantized models exhibit higher sampling entropy—and provides the mechanistic justification for why AQN is beneficial rather than harmful.

- **Fourth**, the **Adaptive Quantization Noise (AQN) mechanism in full detail** (Equations 7, 8, 9, 10, Figure 6), which transforms static quantization error into a dynamic, schedule-controlled exploration signal without adding memory overhead or breaking kernel compatibility.

- **Fifth**, the **Noise Merging technique** (Equation 9, 10, Appendix G proof), which is the implementation trick that folds the injected noise vector into the RMSNorm parameters, converting additive Gaussian noise into row-wise multiplicative noise applied to the quantized weight matrices.

- **Sixth**, the **RL training algorithms and their integration with QeRL** (Equations 3, 4, Algorithm 1), which specifies how GRPO and DAPO operate within the quantized framework, including the role of the reference model, the clipping mechanism, the advantage normalization, and the stage-based noise scheduling.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and empirical analysis paper** whose core idea is that 4-bit quantization, when combined with a dynamically scheduled noise-injection mechanism, simultaneously accelerates LLM reinforcement learning training and enhances exploration, enabling quantized models to outperform their 16-bit counterparts in both training speed and final reasoning accuracy.

---

#### NVFP4 Quantization: The Dual-Scaling 4-Bit Floating-Point Format

The foundation of QeRL's efficiency is the NVFP4 weight format, a 4-bit floating-point representation introduced with NVIDIA's Blackwell GPU architecture and supported on Hopper GPUs (H100) via the Marlin kernel. Understanding NVFP4 requires first understanding the distinction between integer quantization and floating-point quantization, and then the specific dual-scaling mechanism that NVFP4 employs.

**Integer quantization background.** Standard integer quantization maps floating-point weights to a discrete set of $2^N$ integer values. The process is formally defined by Equation 1:

$$\tilde{W} = \text{Round}\left(\frac{W}{s_w}\right), \quad s_w = \frac{W_{\max} - W_{\min}}{q_{\max}}$$

where $W \in \mathbb{R}^{d \times k}$ is the original full-precision weight matrix, $\tilde{W}$ is the quantized integer matrix, $s_w$ is a single per-tensor scaling factor, and $q_{\max} = 2^N - 1$ defines the integer range (e.g., $q_{\max} = 15$ for 4-bit). Dequantization reverses this: $\hat{W} = s_w \cdot \tilde{W}$.

**What this computes:** a uniform quantization grid where each weight value is snapped to the nearest of 16 equally-spaced levels between $W_{\min}$ and $W_{\max}$. The scaling factor $s_w$ encodes the granularity of the grid, and the rounding operation introduces the quantization error $\Delta\epsilon = \hat{W} - W$.

**Why this form:** uniform quantization is hardware-friendly (multiplication with a single scalar factor) but suffers on weight distributions that are not uniform—common in LLMs where weights are approximately normally distributed with outliers. The single scalar factor forces the same precision on dense central regions and sparse tail regions, wasting representational capacity.

**Floating-point quantization and NVFP4's E2M1 format.** Floating-point quantization uses a different quantization grid: values are represented as $\text{mantissa} \times 2^{\text{exponent}}$, which provides higher density near zero (where most weights cluster) and sparser coverage in the tails. NVFP4 uses the E2M1 format: 2 exponent bits and 1 mantissa bit, with $q_{\max} = 6$ (the maximum representable value before scaling). The quantization levels are: $0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0$ (after accounting for the sign bit and special values).

**The dual-scaling mechanism.** What distinguishes NVFP4 from simple FP4 is its hierarchical scaling, formalized in Equation 6 as the dequantization operation:

$$\hat{W} = \text{Dequant}(\tilde{W}) = S_{\text{FP32}} \cdot (S_{\text{E4M3}} \odot \tilde{W})$$

where $S_{\text{FP32}} \in \mathbb{R}$ is a single per-tensor global scaling factor stored in full FP32 precision, $S_{\text{E4M3}}$ is a tensor of block-wise scalers stored in FP8 (E4M3) format—one scaler per block of 16 contiguous weight elements—and $\odot$ denotes block-wise scalar multiplication that broadcasts each scaler to its corresponding 16-element block in the 4-bit weight matrix $\tilde{W}$.

**What this computes:** a two-level scaling hierarchy. The coarse global scaler $S_{\text{FP32}}$ captures the overall magnitude of the weight matrix, ensuring the 4-bit values fall within the representable range. The fine-grained block-wise scalers $S_{\text{E4M3}}$ capture local variations in weight magnitude within each 16-element block, applying per-block adjustments that compensate for outlier channels or uneven activation patterns. The dequantized weight $\hat{W}$ is the high-precision reconstruction used in actual computation.

**Why this dual-scaling form:** a single global scaler (as in integer quantization) forces the same precision on all blocks, which is problematic because different channels in transformer attention and FFN layers often have dramatically different magnitude scales. By adding a second level of block-wise scaling at FP8 precision, NVFP4 achieves finer-grained representation—blocks with large weights get proportionally larger scalers, preserving their relative precision, while blocks with small weights get smaller scalers that prevent underflow. This is the same principle behind block-wise quantization methods like AWQ (Lin et al., 2024) and GPTQ (Frantar et al., 2022), but implemented in a floating-point format with native hardware support via Marlin kernels.

**Comparison to other 4-bit formats studied in the paper:**

- **MXFP4** (Microscaling FP4, Project 2023): Also uses E2M1 format but with a shared FP8 (E8M0) scaling factor across blocks of 32 elements—larger block size means coarser scaling granularity. The paper finds MXFP4 achieves competitive early-stage training rewards but converges to lower final rewards than NVFP4 (Figure 4), suggesting finer-grained scaling matters for retaining the weight structure that enables effective RL optimization.

- **NF4** (NormalFloat 4-bit, Dettmers et al., 2023a): Uses data-dependent quantization levels optimized for normally distributed weights. The levels are placed at quantiles of the Gaussian distribution rather than at uniform or floating-point spacings. This format works well for representing pretrained weights but requires a lookup table to map 4-bit indices to floating-point values during computation—the unpacking overhead that makes QLoRA 1.5-2× *slower* than BF16 (Table 3). NF4's slower reward growth in Figure 4 likely reflects both this speed penalty and the poorer compatibility of its quantization grid with the weight perturbation patterns that drive exploration in RL.

**The Marlin kernel acceleration.** NVFP4's practical advantage over NF4 stems from hardware support: the Marlin kernel (Frantar et al., 2024) implements NVFP4 × BF16 matrix multiplication directly on H100 tensor cores without unpacking. The kernel accepts 4-bit NVFP4 weights and BF16 activations as inputs, performs the dual-scaling dequantization and matrix multiplication in a single fused operation, and outputs BF16 results. This eliminates the memory bandwidth bottleneck of loading unpacked FP16 weights and avoids the lookup table indirection that NF4 requires. The paper's speedup measurements in Section 4.3 quantify the result: QeRL achieves 1.2-1.5× end-to-end training speedup over BF16 LoRA and 1.8-2.0× over QLoRA, with the advantage growing larger for bigger models (2.0× speedup for the 32B model at batch size 8, Table 8).

**Quantization-aware weight calibration.** Before training, the pretrained BF16 model weights are quantized to NVFP4 using AWQ (Activation-aware Weight Quantization, Lin et al., 2024). This is not naive rounding but a calibration process that uses a small set of calibration data—the paper uses 256 sequences of 2048 tokens each, sampled from OpenThoughts-114k (Guha et al., 2025)—to determine per-channel scaling factors that minimize the error between the original and quantized model's outputs on representative inputs. The calibration identifies channels where quantization error disproportionately affects the model's output (due to activation outliers) and allocates more scaling precision to those channels, preserving accuracy while maintaining the 4-bit format.

---

#### Low-Rank Adaptation (LoRA) Within Quantized Layers

QeRL employs LoRA (Hu et al., 2022) as the parameter-efficient fine-tuning mechanism, applied on top of the frozen NVFP4-quantized base weights. The core idea of LoRA is that weight updates during fine-tuning lie in a low-dimensional subspace, meaning the full-rank update matrix $\Delta W \in \mathbb{R}^{d \times k}$ can be approximated by a low-rank factorization. This is formalized in Equation 2:

$$W + \Delta W = W + BA$$

where $W \in \mathbb{R}^{d \times k}$ is the frozen pretrained weight matrix (in QeRL, the NVFP4-quantized weight $\hat{W}$), $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ are trainable low-rank matrices, and $r \ll \min(d, k)$ is the rank—typically 16 to 128 in the paper's experiments, with $r = 32$ used as the default in main experiments.

**What this computes:** the effective weight during the forward pass is the sum of the frozen quantized weight and the low-rank update. The forward computation for a linear layer with input $x$ becomes:

$$y = x \cdot (\hat{W} + BA) = x \cdot \hat{W} + x \cdot BA$$

The first term $x \cdot \hat{W}$ uses the Marlin-accelerated NVFP4 × BF16 kernel. The second term $x \cdot BA$ is computed in standard BF16 precision (since $A$ and $B$ are stored and updated in BF16). The results are summed to produce the layer output.

**Why this form:** the decomposition separates the memory-heavy frozen weights (which benefit from 4-bit compression and hardware acceleration) from the lightweight trainable adapters (which need full precision for gradient computation). The rank $r$ controls the expressivity-memory tradeoff: higher ranks capture more complex adaptations but require more memory and computation. The paper's rank ablation (Figure 10) shows that ranks 16 through 128 exhibit similar reward growth rates and convergence behavior for the 3B model, with rank 16 converging slightly faster—suggesting that the exploration benefit from quantization noise reduces the need for high adapter expressivity, since the noise itself encourages the model to explore diverse weight configurations implicitly.

**Layer coverage.** QeRL applies LoRA to the full set of linear projection matrices in each transformer block: the attention projections ($W_q, W_k, W_v, W_o$) and the feed-forward network projections ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). This mirrors the standard LoRA application pattern in LLM fine-tuning, covering all layers that contribute to the model's representational capacity. The paper notes that only about 1% of the total parameters are trainable (the LoRA matrices plus RMSNorm parameters), yet the model achieves performance matching full-parameter fine-tuning on mathematical reasoning benchmarks (Table 2).

**Gradient flow.** During backpropagation, gradients flow only through the $A$ and $B$ matrices. The quantized base weights $\hat{W}$ are completely frozen and receive no gradient updates. This has two consequences: (1) the optimizer state memory (AdamW moments) is proportional only to the number of trainable LoRA parameters, not the full model size—a key memory savings; (2) the quantization noise pattern $\Delta\epsilon = \hat{W} - W$ remains constant throughout training, which is both the source of the static noise limitation that AQN addresses and the reason why the exploration behavior is consistent and analyzable.

**Interaction with AQN noise.** The AQN mechanism (detailed below) injects noise before the quantized linear layers by modifying the RMSNorm scaling factors. Since LoRA adapters receive the same noisy activations as the quantized base weights, the noise affects both the $\hat{W}$ path and the $BA$ path equally during the forward pass. This means the policy gradient updates to $A$ and $B$ must learn to produce effective reasoning outputs under noisy conditions—a form of implicit regularization that may contribute to the observed robustness to higher learning rates (Figure 16, where QeRL tolerates learning rate $3 \times 10^{-5}$ while BF16 LoRA collapses at this rate).

---

#### Quantization Noise as an Exploration Mechanism: The Central Empirical Finding

The paper's most intellectually distinctive contribution is the discovery and mechanistic explanation of how quantization noise enhances RL exploration. This is not an engineering optimization but an empirical finding that challenges the conventional SFT-era view that quantization degrades training.

**The observation.** When applying PEFT-based RL (LoRA on quantized weights) to mathematical reasoning tasks, the authors observe that 4-bit quantized models consistently achieve:
- Faster reward convergence (steeper upward slope in training reward curves, Figure 4)
- Higher final evaluation accuracy (Table 1: NVFP4+LoRA achieves 88.5 on GSM8K vs. 88.1 for BF16 LoRA on Qwen2.5-7B-Instruct)
- Better performance than their 16-bit LoRA counterparts across model sizes (3B, 7B, 14B, 32B) and datasets (GSM8K, BigMath)

This is surprising because in SFT, quantization uniformly degrades performance. Table 1 confirms this degradation for the base quantized models (without RL): NVFP4 quantization alone drops GSM8K accuracy from 76.3 to 73.4 for the 7B model, and from 61.2 to 59.4 for the 3B model. The fact that RL training reverses this deficit—NVFP4+LoRA+RL surpasses BF16+LoRA+RL—indicates a mechanism specific to RL that benefits from quantization.

**The mechanism: quantization noise increases sampling entropy.** The paper provides a formal model of how quantization affects the policy's output distribution. For a given input query $q$, the policy $\pi_\theta$ produces a probability distribution over the vocabulary $V$ at each token position $t$:

$$H(\pi(\cdot|q)) = -\sum_{o_t \in V} \pi(o_t|q) \log \pi(o_t|q)$$

where $H$ is the Shannon entropy measured in nats, $\pi(o_t|q)$ is the probability assigned to token $o_t$ given the query $q$ and the preceding context, and the sum runs over all tokens in the vocabulary $V$.

**What this computes:** a measure of the "flatness" or uncertainty of the output distribution. High entropy means probability mass is spread across many tokens (the model considers many alternatives plausible); low entropy means mass is concentrated on a few high-confidence tokens. In the context of autoregressive generation, entropy at each step determines how diverse the model's completions will be.

The paper's key empirical finding, illustrated in Figures 5 and 14, is that quantized models exhibit **consistently higher entropy** than their 16-bit counterparts throughout RL training. Figure 5 shows this quantitatively: NVFP4-LoRA (rank 32) maintains entropy around 0.22-0.28 across training steps, while BF16-LoRA (rank 32) entropy ranges around 0.15-0.18. This difference—approximately 0.07-0.10 nats higher entropy—persists even as both models' entropy evolves during training. Figure 14 extends this finding to the larger 14B model, where QeRL entropy remains visibly above the LoRA baseline throughout 600 training steps.

**Why quantization increases entropy.** The causal chain from quantization to entropy is modeled through Equation 5, which describes the effective perturbation to the model parameters:

$$(\tilde{\theta} + \theta_{\text{lora}}) - (\theta + \theta_{\text{lora}}) = Q(\theta) - \theta = \Delta\epsilon$$

where $\tilde{\theta}$ represents the dequantized weights (the NVFP4 weights reconstructed back to high precision via Equation 6), $\theta$ represents the original full-precision weights, $\theta_{\text{lora}}$ represents the LoRA adapter parameters (identical in both cases since they sit on top of the base weights), $Q(\theta)$ is the quantization function, and $\Delta\epsilon$ is the resulting quantization error—the per-weight difference between the quantized reconstruction and the original value.

**What this computes:** the effective perturbation experienced by the model's computations compared to the ideal full-precision computation. The quantization error $\Delta\epsilon$ is a structured, weight-dependent noise pattern: it is not random Gaussian noise but a deterministic function of the weight distribution, the quantization grid, and the block-wise scaling factors. For a typical LLM weight matrix, $\Delta\epsilon$ has the following properties: (1) it is small relative to the weight magnitude (4-bit quantization preserves ~6-7 bits of effective precision), (2) it is systematic—the same weight always maps to the same quantized value, producing the same error, and (3) it propagates nonlinearly through the network layers.

When the quantized model performs a forward pass, each layer's computation introduces this small systematic error into the activations. The error propagates through subsequent layers, where it interacts with nonlinearities (GELU activations, softmax normalization, residual connections) and accumulates. By the final logit layer—just before the softmax that produces the token probability distribution—the accumulated perturbations have "flattened" the logit vector slightly: high logits are reduced, low logits are increased, but the ordering is largely preserved. The softmax then amplifies this flattening effect: $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$, and small reductions in peak logits produce disproportionately large reductions in the corresponding token probabilities, redistributing that probability mass across other tokens.

The result is the entropy increase observed in Figure 5. The output distribution becomes less peaked, assigning more meaningful (non-negligible) probabilities to a wider range of tokens. This is visually illustrated in Figure 3: the probability distribution with quantization (left panel) is noticeably flatter than the distribution without quantization (right panel), with the peak probability reduced and the tail probabilities elevated.

**Why higher entropy benefits RL.** The connection to reinforcement learning is through the exploration-exploitation trade-off. In RL for LLM reasoning, the model must generate diverse candidate solutions to discover which reasoning strategies lead to correct answers. If the model is overconfident—assigning nearly all probability mass to its initial instinct about the next token—it will produce a narrow set of similar completions, missing alternative strategies that might be more effective. Higher entropy during sampling means the model is more likely to try different reasoning paths, different intermediate steps, and different final approaches.

This is analogous to parameter-space noise exploration in traditional RL (Plappert et al., 2017), where Gaussian noise is deliberately added to the policy network's weights to induce varied behavior. The key difference is that in QeRL, this exploration noise emerges "for free" from quantization—no explicit noise generation infrastructure is needed because the quantization error itself serves as a perturbation mechanism. Equation 5 formalizes this: the term $\Delta\epsilon = Q(\theta) - \theta$ is precisely the analog of the injected parameter noise $\mathcal{N}(0, \sigma^2)$ in parameter-noise methods, but it is deterministic (for a given quantization configuration) rather than randomly sampled.

The paper's ablation of quantization formats (Figure 4) provides indirect evidence for this mechanism. NVFP4 and MXFP4—both floating-point formats that provide finer-grained precision near zero where most weights cluster—exhibit better reward growth than NF4, which uses a data-dependent quantization grid. This suggests that the *pattern* of quantization noise (which weights get perturbed and by how much) matters for exploration effectiveness, not just the average noise magnitude. NVFP4's dual-scaling mechanism produces a noise pattern that is correlated with weight magnitude and block-level statistics, potentially providing a more structured exploration signal that preserves the model's core reasoning capabilities while perturbing less critical parameters.

**The critical limitation: static noise is insufficient for dynamic RL training.** While the inherent quantization noise provides a useful baseline exploration signal, the paper identifies a fundamental mismatch with RL's requirements. In standard RL with explicit noise injection, the noise is:
- **Randomly sampled** at each training step (or each episode), providing varied perturbations that explore different regions of parameter space.
- **Schedule-controlled**, with noise magnitude typically decaying from high (encouraging broad exploration early) to low (allowing precise exploitation of learned strategies late).

Quantization noise has neither property. The error pattern $\Delta\epsilon$ is fixed once the weights are quantized—every forward pass experiences the same perturbation pattern. This means the model explores a single fixed "shifted" version of its parameter space, potentially missing strategies that would be accessible under different perturbation patterns. Moreover, the noise magnitude is also fixed, providing the same exploration pressure throughout training regardless of whether the model needs it (early) or would benefit from reduced variance (late).

This limitation motivates the core technical novelty of QeRL: the Adaptive Quantization Noise (AQN) mechanism, which overlays a dynamically controlled stochastic noise component on top of the static quantization error.

---

#### Adaptive Quantization Noise (AQN): Dynamic Exploration Control

The AQN mechanism transforms the static, deterministic quantization noise into a dynamic, schedule-controlled exploration signal. It consists of three integrated components: (1) a stochastic noise vector sampled independently for each forward pass, (2) an exponential decay schedule that adjusts the noise magnitude across training stages, and (3) a noise-merging technique that folds the noise into the RMSNorm parameters to avoid memory overhead and kernel compatibility issues.

**The stochastic noise vector.** For each quantized linear layer in the model, AQN samples an additive Gaussian noise vector before the forward pass. The formal definition of the augmented noise is given in Equation 7:

$$\Delta\epsilon' = Z_{\text{noisy}} + \Delta\epsilon = Z_{\text{noisy}} + (\hat{W} - W)$$

where $Z_{\text{noisy}} = \epsilon$, with $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$, is a random vector of dimension $1 \times d$ (matching the input dimension $d$ of the linear layer), each element drawn independently from a zero-mean Gaussian with standard deviation $\sigma$, $\Delta\epsilon = \hat{W} - W$ is the static quantization error from Equation 5, and $\Delta\epsilon'$ is the total effective perturbation applied to the layer's computation.

**What this computes:** the sum of two noise sources—the fixed quantization error $\Delta\epsilon$ (always present, providing a baseline exploration floor) and the freshly sampled stochastic noise $Z_{\text{noisy}}$ (providing step-to-step variation and schedule-controllable magnitude). The stochastic component is independently resampled for every forward pass, meaning each rollout generation experiences a different perturbation pattern even though the underlying quantized weights remain frozen.

**Why additive Gaussian noise:** the choice of Gaussian noise follows the parameter-space exploration literature (Plappert et al., 2017; Fortunato et al., 2018), where additive Gaussian perturbations to network parameters have been shown to induce effective exploration in continuous control RL. The Gaussian distribution provides two desirable properties: (1) it is isotropic (directionally unbiased), meaning the perturbation explores all directions in parameter space equally in expectation, and (2) the $\sigma$ parameter provides a single scalar knob to control the exploration-exploitation trade-off—larger $\sigma$ encourages more diverse behavior, smaller $\sigma$ allows more precise execution of learned strategies.

The paper specifies the noise dimension as $1 \times d$ (a row vector of length equal to the input dimension of the linear layer). This means the noise is injected per-layer, not per-weight: a single noise vector perturbs all output channels of the layer uniformly. This is a design choice that balances exploration granularity against parameter overhead—injecting per-weight noise (a $d \times k$ matrix) would require storing a noise tensor as large as the weight matrix itself, defeating the memory efficiency purpose of quantization.

**The exponential decay schedule.** The noise standard deviation $\sigma$ is not constant throughout training but follows an exponential decay across $K$ evenly-spaced stages. The schedule is defined by Equation 8:

$$\sigma(k) = \sigma_{\text{start}} \cdot \left(\frac{\sigma_{\text{end}}}{\sigma_{\text{start}}}\right)^{\frac{k-1}{K-1}}$$

where $\sigma_{\text{start}} = 1 \times 10^{-2}$ is the initial noise level at the first active noise stage ($k = 1$), $\sigma_{\text{end}} = 5 \times 10^{-4}$ is the final noise level at the last stage ($k = K$), $k \in \{1, 2, \dots, K\}$ indexes the current training stage, and $K$ is the total number of stages—set to 10 for the paper's GSM8K experiments (with approximately 600 total training steps, meaning each stage spans roughly 60 steps).

**What this computes:** the noise standard deviation for the current training stage, interpolating geometrically between $\sigma_{\text{start}}$ and $\sigma_{\text{end}}$. At stage $k = 1$, $\sigma = 10^{-2}$ (relatively high noise, strong exploration). At stage $k = K$, $\sigma = 5 \times 10^{-4}$ (low noise, exploitation-dominant). At intermediate stages, $\sigma$ follows geometric interpolation, meaning the ratio between successive stages is constant: $\sigma(k+1) / \sigma(k) = (\sigma_{\text{end}} / \sigma_{\text{start}})^{1/(K-1)}$.

**Why exponential decay:** the paper compares four decay schedules—linear, exponential, cosine, and logarithmic (Figure 9, Figure 15)—and selects exponential decay because it "achieves more stable improvements later by reducing noise to lower levels." To understand why, consider the shape of each schedule:

- **Linear decay:** $\sigma(k) = \sigma_{\text{start}} - (\sigma_{\text{start}} - \sigma_{\text{end}}) \cdot (k-1)/(K-1)$. Noise decreases by a constant absolute amount each stage, meaning early stages see small relative reductions while late stages see large relative reductions.
- **Exponential decay:** constant relative reduction per stage (Equation 8). This means noise drops rapidly at the beginning (when $\sigma$ is large, a constant factor reduction produces a large absolute drop) and more gradually at the end (when $\sigma$ is small, the same factor reduction produces a small absolute drop), maintaining a low but non-zero exploration pressure in late training.
- **Cosine decay:** $\sigma(k) = \sigma_{\text{end}} + \frac{1}{2}(\sigma_{\text{start}} - \sigma_{\text{end}})(1 + \cos(\pi \cdot (k-1)/(K-1)))$. Smooth S-shaped curve that stays near $\sigma_{\text{start}}$ for early stages and near $\sigma_{\text{end}}$ for late stages.
- **Logarithmic decay:** $\sigma(k) = \sigma_{\text{start}} - (\sigma_{\text{start}} - \sigma_{\text{end}}) \cdot \log(k) / \log(K)$. Very sharp initial drop, then long tail.

The exponential schedule's advantage (Figure 9) likely stems from its ability to maintain a meaningful exploration signal through the middle stages of training. In linear decay, noise becomes very small by stage 5-6 (50% of training), potentially causing premature convergence to suboptimal strategies before the model has fully explored the solution space. The exponential schedule keeps noise at moderate levels through more of training, then rapidly drops to near-zero in the final stages, allowing fine-tuning of the discovered strategies. The cosine schedule's initial plateau may waste early training steps with unnecessarily high noise, while the logarithmic schedule's sharp drop may cut off exploration too early.

**Stage 0: pure quantization noise.** The paper's Algorithm 1 specifies an important detail: at stage $k = 0$ (the initial training stage before AQN activates), $\sigma = 0$, meaning only the inherent quantization noise $\Delta\epsilon$ is present. This stage allows the model to begin adapting its LoRA parameters under the baseline quantization perturbation before the additional stochastic noise is introduced. The paper does not explicitly state the number of steps in stage 0, but Algorithm 1 indicates that the total training steps $M$ are divided into $K$ equal stages, and the noise level depends on which stage the current step falls into.

**Ablation evidence.** Figure 8 provides the key evidence that AQN improves over static quantization noise. Training the 3B model with AQN (orange curve) versus without AQN (blue curve) shows:
- Early training (steps 0-20): both curves overlap, confirming that stage 0 (pure quantization noise) provides sufficient exploration for initial learning.
- Mid training (steps 20-60): the AQN curve diverges upward, showing that the injected stochastic noise helps the model discover better strategies than static noise alone.
- Late training (steps 60+): the AQN curve continues to rise while the non-AQN curve plateaus, and notably, AQN "effectively expands the model's exploration space, enabling further improvements in reward" near convergence.

This pattern supports the paper's claim that static quantization noise, while beneficial early, becomes a limitation later—the model settles into a local optimum determined by the fixed perturbation pattern, and AQN's dynamic noise injects the variability needed to escape and discover higher-reward strategies.

**Noise injection frequency.** Algorithm 1 specifies that the noise level $\sigma$ is set once per training step (line 7), and the stochastic noise vector $Z_{\text{noisy}}$ is resampled for each forward pass (implied by the AQN mechanism resampling $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ for each forward pass). This means that within a single training step:
- All $G$ rollouts for a given prompt are generated with the same $\sigma$ (since the noise level depends on the training step, not the rollout index), but each rollout experiences a different sampled $\epsilon$ (since $Z_{\text{noisy}}$ is resampled per forward pass).
- The log-probability evaluation for policy gradient computation uses the same quantized weights but may be performed without the explicit AQN noise injection (the paper does not fully specify whether AQN noise is applied during log-probability evaluation or only during rollouts—this is a detail that would affect the on-policy nature of the updates).

---

#### Noise Merging: Zero-Parameter-Overhead Noise Injection via RMSNorm

A practical challenge arises when trying to inject the noise vector $Z_{\text{noisy}}$ into a quantized linear layer: one cannot simply add a high-precision noise vector to the 4-bit quantized weights, because the weights are stored in NVFP4 format and the Marlin kernel expects to perform NVFP4 × BF16 multiplication without intermediate dequantization into a general-purpose floating-point buffer. Explicitly creating a separate noise vector for each layer would also impose a parameter-efficiency burden, increasing memory overhead proportional to the total number of input dimensions across all layers.

The paper resolves both issues with an elegant equivalency transformation, formalized in Equation 9:

$$X \cdot (Z_{\text{noisy}} + \hat{W}) = X \cdot Z_{\text{noisy}} + X \cdot \hat{W}$$

where $X \in \mathbb{R}^{n \times d}$ is the input activation matrix (with batch size $n$ and feature dimension $d$), $Z_{\text{noisy}} \in \mathbb{R}^{1 \times d}$ is the row-vector noise term (broadcast across the batch dimension during multiplication), and $\hat{W} \in \mathbb{R}^{d \times k}$ is the dequantized weight matrix.

**What this computes:** the equivalence shows that adding noise to the weights is mathematically identical to adding a noise-dependent bias term to the activations before the linear transformation. The left side represents the conceptual operation—perturb the weights and then multiply by activations. The right side represents the implementable operation—multiply activations by the clean quantized weights (which the Marlin kernel can handle) and add the noise contribution $X \cdot Z_{\text{noisy}}$ as a separate term (which can be computed in BF16 using standard matrix operations).

**The RMSNorm integration.** The paper goes one step further by observing that $X \cdot Z_{\text{noisy}}$ can be absorbed into the preceding normalization layer. In transformer architectures, each linear layer's input comes from either a LayerNorm/RMSNorm operation or from the residual stream after normalization. By folding the noise vector into the RMSNorm's learnable scale parameter $w$, the noise injection becomes completely transparent to the downstream linear layer—the Marlin kernel sees only the normalized (and now noisily scaled) activations multiplied by the clean NVFP4 weights. This is formalized in Equation 10:

$$\text{RMSNorm}_{\text{noise}}(x) = w_{\text{noise}} \odot \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \delta}}, \quad w_{\text{noise}} = Z_{\text{noise}} + w$$

where $x \in \mathbb{R}^N$ is the input vector to the RMSNorm operation (with $N$ being the hidden dimension), $\frac{x}{\sqrt{\frac{1}{N} \sum x_i^2 + \delta}}$ is the standard RMSNorm normalization (divide by root-mean-square with a small constant $\delta$ for numerical stability), $w \in \mathbb{R}^N$ is the original learnable scaling factor of RMSNorm, $Z_{\text{noise}} \in \mathbb{R}^{1 \times d}$ is the injected noise vector (broadcast to match the $N$ dimension), and $w_{\text{noise}}$ is the effective scaling factor used for the forward pass—the element-wise sum of the original scale and the noise.

**What this computes:** a modified RMSNorm that scales each normalized feature dimension by $(w_i + Z_{\text{noisy},i})$, where $w_i$ is the learned scaling factor and $Z_{\text{noisy},i}$ is the noise perturbation for that dimension. The rest of the RMSNorm operation (the mean-subtraction-like normalization step) remains unchanged. The output is a normalized activation vector where each feature dimension has been scaled by a slightly perturbed factor.

**Why this approach works:** the RMSNorm operation is applied element-wise to each feature dimension, meaning the scaling factor $w$ multiplies each normalized feature independently. Adding noise to $w$ is therefore equivalent to applying multiplicative noise to each feature dimension. The downstream linear layer then multiplies these noisily-scaled features by the quantized weight matrix $\hat{W}$. Through the distributive property of matrix multiplication, this is equivalent to having added a noise-dependent bias to the activations before the linear layer—exactly the $X \cdot Z_{\text{noisy}}$ term from Equation 9.

**Conversion from additive to multiplicative noise.** Appendix G provides the proof that the noise-sharing operation in Equation 10 transforms the additive Gaussian noise into multiplicative noise applied to the weight matrix. The derivation proceeds as follows (Equation 11):

$$\text{RMSNorm}_{\text{noise}}(X) = \left(\frac{Z_{\text{noise}}}{w} + I\right) \odot \text{RMSNorm}(X)$$

where $I$ is the identity matrix (all ones), $\text{RMSNorm}(X)$ is the standard normalized output, and $\frac{Z_{\text{noise}}}{w}$ represents element-wise division of the noise by the learned scale. This shows that the effective operation is a multiplicative perturbation by the factor $(\frac{Z_{\text{noise}}}{w} + I)$ applied to the normalized activations.

Then, when these perturbed activations are multiplied by the weight matrix (Equation 12):

$$\left(\left(\frac{Z_{\text{noise}}}{w} + I\right) \odot \text{RMSNorm}(X)\right) \cdot \hat{W} = \text{RMSNorm}(X) \cdot \left(\left(\frac{Z_{\text{noise}}}{w} + I\right)^\top \odot \hat{W}\right)$$

**What this equation shows:** the multiplicative noise on the activations can be equivalently viewed as row-wise multiplicative noise applied to the weight matrix $\hat{W}$. Specifically, each row $j$ of $\hat{W}$ is multiplied by $(Z_{\text{noisy},j} / w_j + 1)$. This is the final form in which the noise actually affects the computation: as a per-output-channel multiplicative perturbation to the quantized weights.

**Why multiplicative noise matters:** the paper notes (citing Pang & Jiang, 2021; Zhang et al., 2025a) that multiplicative noise has been shown to be effective in RL, particularly for robust control. However, it also notes that multiplicative noise is "more sensitive, especially in deep networks like LLMs," which is why the initial noise standard deviation $\sigma_{\text{start}} = 1 \times 10^{-2}$ is set smaller than the typical $\sigma_{\text{start}} = 0.1$ used in traditional noise-injection networks. The smaller initialization prevents the multiplicative perturbations from destabilizing the early training dynamics when the LoRA adapters are still randomly initialized.

**Per-block noise sharing.** The paper exploits the architecture of standard transformer blocks to minimize the number of independent noise vectors needed. Specifically (as shown in Figure 6):
- In the multi-head self-attention block, the $Q, K, V$ projections share the same RMSNorm and therefore the same noise vector $Z_{\text{noise}}$. The output projection $W_o$ has its own noise vector.
- In the feed-forward network block, the gate and up projections ($W_{\text{gate}}, W_{\text{up}}$) share the same RMSNorm and therefore the same noise vector. The down projection ($W_{\text{down}}$) has its own noise vector.

This noise-sharing scheme means that for each transformer layer, only 3 independent noise vectors are needed (QKV group, output projection, gate-up group), rather than one per linear layer. This keeps the total noise injection parameter count negligible—zero additional storage since the noise is folded into existing RMSNorm parameters—while still providing distinct exploration perturbations for different functional components of the transformer.

**Kernel compatibility.** A critical constraint that the noise merging satisfies is that the Marlin kernel for NVFP4 × BF16 multiplication expects clean NVFP4 weights and BF16 activations as inputs. If noise were added directly to the dequantized weights (the left side of Equation 9), the multiplication would need to handle a dynamic component that changes every forward pass—breaking the ability to use the optimized, pre-compiled kernel. By moving the noise into the RMSNorm parameters (which are standard BF16 tensors), the Marlin kernel sees only the standard interface: quantized weights $\hat{W}$ and BF16 activations (the noisily-scaled RMSNorm outputs). The noise structure is completely transparent to the kernel, preserving full hardware acceleration.

---

#### Training Algorithms: GRPO and DAPO Integration with QeRL

QeRL is designed to work with modern policy optimization algorithms for LLM reasoning, specifically GRPO (Group Relative Policy Optimization, Shao et al., 2024) and DAPO (Dynamic Sampling Policy Optimization, Yu et al., 2025). The framework wraps these algorithms with the quantized backbone, LoRA adapters, and AQN noise injection described above, as detailed in Algorithm 1.

**Group Relative Policy Optimization (GRPO).** GRPO is a variant of policy gradient methods that eliminates the need for a separately trained reward model (as required in PPO) by using group-relative advantage estimation. For a given input query $q$, the model generates $G$ candidate outputs $\{o_1, o_2, \dots, o_G\}$ (8 for GSM8K, 16 for BigMath), each scored by a rule-based reward function $r_\phi$ (exact-answer matching against the ground truth). The optimization objective is given in Equation 3:

$$J(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( \frac{\pi_\theta(o_{i,t}|q)}{\pi_{\theta_{\text{old}}}(o_{i,t}|q)} A_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t}|q)}{\pi_{\theta_{\text{old}}}(o_{i,t}|q)}, 1 - \alpha, 1 + \alpha \right) A_{i,t} \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

where:
- $q$ is a query sampled from the training dataset $D$,
- $\{o_i\}_{i=1}^G$ are the $G$ sampled outputs for query $q$,
- $|o_i|$ is the number of tokens in output $o_i$,
- $\pi_\theta(o_{i,t}|q)$ is the probability the current policy assigns to token $o_{i,t}$ given the query and previous tokens,
- $\pi_{\theta_{\text{old}}}(o_{i,t}|q)$ is the probability under the old policy (frozen at the start of the current update step),
- $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ is the importance sampling ratio—how much the current policy's probability for token $o_{i,t}$ differs from the old policy's probability,
- $A_{i,t}$ is the advantage estimate for token $t$ in output $i$ (defined in Equation 4),
- $\text{clip}(\cdot, 1-\alpha, 1+\alpha)$ restricts the importance ratio to the interval $[1-\alpha, 1+\alpha]$ (with $\alpha = 0.2$ per Table 4), preventing any single update from changing the policy too drastically,
- $\min(\cdot, \cdot)$ takes the more conservative of the unclipped and clipped objectives—this is the standard PPO-style pessimistic clipping that prevents the policy from moving toward actions where the advantage estimate is unreliable due to large importance ratios,
- $\beta$ is the KL penalty coefficient,
- $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ is the Kullback-Leibler divergence between the current policy and a reference policy (usually the initial pretrained model), which penalizes the policy for diverging too far from reasonable text distributions.

**What this computes:** the expected per-token policy gradient update. For each token in each generated output, the objective pushes the policy to increase the probability of that token if the advantage $A_{i,t}$ is positive (the output was better than average for its query group), and decrease it if the advantage is negative. The clipping and minimum operations ensure the update is conservative—the policy does not change too much in a single step, which would risk destroying previously learned reasoning capabilities. The KL penalty provides an additional regularizer keeping the policy close to the reference model.

**Why group-relative advantages:** rather than using an absolute reward signal (which requires a trained reward model that generalizes across diverse queries), GRPO normalizes rewards within each query's group of $G$ outputs. The rationale is that for mathematical reasoning tasks, the absolute difficulty of different queries varies enormously (some are easy, some impossible), but the relative quality of different solution attempts for the *same* query can be compared meaningfully without cross-query calibration.

**The advantage computation** (Equation 4):

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$

where $r_i \in \{0, 1\}$ is the binary reward for output $o_i$ (1 if the extracted answer matches the ground truth, 0 otherwise), $\text{mean}(\{r_1, \dots, r_G\})$ is the average reward across all $G$ outputs for this query, and $\text{std}(\{r_1, \dots, r_G\})$ is the standard deviation of these rewards.

**What this computes:** a z-score normalized advantage for each output. If output $i$ received a reward of 1 while the group average is 0.3, its advantage is positive $(1 - 0.3) / \text{std}$, encouraging the policy to produce tokens like those in output $i$ more often. If output $j$ received 0 while the average is 0.3, its advantage is negative, discouraging similar tokens. The shared advantage $A_i$ is applied to every token in output $o_i$—a simplification justified by the fact that mathematical reasoning tasks have sparse rewards (only the final answer matters), making per-token credit assignment difficult.

**Why this normalization:** the division by standard deviation ensures that the scale of policy updates is consistent across queries regardless of how varied the output quality is. For queries where all outputs are wrong ($r_i = 0$ for all $i$), the standard deviation is 0, which would cause a division-by-zero. The paper handles this edge case by setting the advantage to 0 when the standard deviation is 0—no update is made when all outputs are equally wrong, since there's no signal about which tokens are better or worse.

**DAPO modifications.** DAPO (Yu et al., 2025) introduces two changes to the GRPO formulation that the paper uses for its BigMath experiments:

1. **Higher clipping upper bound.** DAPO increases the upper clipping bound $\epsilon_{\text{high}}$ from the standard 0.2 to 0.28 (Table 4: "Clip range $\epsilon_{\text{low}}, \epsilon_{\text{high}}$: 0.2, 0.28"). This allows larger policy updates in the positive direction (increasing probabilities of high-advantage tokens) while still restricting negative updates (decreasing probabilities of low-advantage tokens). The motivation is to avoid **entropy collapse**—a phenomenon where the policy becomes overly deterministic and loses the exploration capacity needed to discover better strategies. By allowing larger upward adjustments, DAPO makes it harder for the policy to collapse to a single mode.

2. **Removal of KL penalty.** DAPO removes the KL divergence term $-\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ from Equation 3 entirely. The KL penalty in standard GRPO/PPO prevents the policy from deviating too far from the reference model, which is important for maintaining fluency and factual accuracy in general language tasks. However, for reasoning tasks where the model needs to explore novel reasoning patterns, the KL penalty acts as an upper limit on exploration—it penalizes the policy for discovering strategies that are effective but distributionally different from the pretrained model's outputs. By removing this penalty, DAPO "eliminates the upper limit on exploration in RL, thereby encouraging more optional tokens in the rollout process."

**Implications for QeRL.** The DAPO modifications are synergistic with QeRL's exploration-enhancement mechanism. QeRL's AQN already increases exploration by perturbing the model's output distribution—the higher clipping bound and removed KL penalty in DAPO ensure that the model can *capitalize* on this exploration by making larger policy updates toward the high-reward strategies that the noise helps discover. This synergy may explain why QeRL shows consistent benefits across both GRPO and DAPO (Figure 4), but particularly strong reward growth with DAPO on the challenging BigMath dataset (Figure 7, where QeRL achieves rapid reward increase within 200 steps vs. 500+ for vanilla LoRA).

**Algorithm 1: QeRL training loop in detail.** The algorithm (pseudocode in Appendix F) describes the complete per-iteration process:

1. **Initialization (line 1):** The policy model $\pi_\theta$ is constructed as $\tilde{\pi}_\theta + \theta_{\text{lora}}$—the NVFP4-quantized base model plus randomly initialized LoRA adapters. The reward function $r_\phi$ is the rule-based answer checker. Hyperparameters include LoRA rank (32 by default), LoRA $\alpha$ (scaling factor), number of stages $K$ (10 for GSM8K), $\sigma_{\text{start}} = 1 \times 10^{-2}$, and $\sigma_{\text{end}} = 5 \times 10^{-4}$.

2. **Reference model snapshot (line 3):** At the start of each outer iteration, the reference model $\pi_{\text{ref}}$ is set to a copy of the current policy. This reference model is used only in the KL penalty term (for GRPO) or not at all (for DAPO).

3. **Stage-based noise scheduling (lines 5-7):** The total training steps $M$ within the iteration are divided into $K$ equal stages (each spanning $\lfloor M/K \rfloor$ steps). For each step, the algorithm determines the current stage $k$ by integer division, then sets $\sigma$ according to the exponential decay formula (Equation 8). At stage $k = 0$, $\sigma = 0$ (pure quantization noise only).

4. **AQN noise injection (line 9):** The old policy model $\pi_{\theta_{\text{old}}}$ is updated with AQN noise: $\pi_{\theta_{\text{old}}} \leftarrow \pi_\theta + \mathcal{N}(0, \sigma^2)$. In practice (via the noise merging technique), this means the RMSNorm scaling parameters are perturbed by the sampled Gaussian noise, which implicitly adds noise to the effective weights as shown in Equation 12.

5. **Rollout generation (line 10):** For each query $q$ in the current batch $D_b$, the noisy policy generates $G$ candidate outputs autoregressively using the Marlin-accelerated NVFP4 backbone. Each output is a complete reasoning trace formatted as `thinking ... response <answer> ... </answer>` to enable answer extraction for reward computation.

6. **Reward computation (line 11):** The rule-based reward function $r_\phi$ extracts the final answer from each output and compares it to the ground truth, yielding binary rewards $r_i \in \{0, 1\}$.

7. **Advantage estimation (line 12):** Group-relative advantages $\hat{A}_{i,t}$ are computed per Equation 4 and assigned to every token in each output.

8. **Inner-loop policy updates (lines 13-15):** The policy model $\pi_\theta$ is updated for $\mu$ inner-loop steps (4 for GSM8K with off-policy updates, 1 for BigMath with on-policy updates) by maximizing the GRPO/DAPO objective (Equation 3) using gradient ascent. Only the LoRA matrices and RMSNorm parameters receive gradient updates—the NVFP4 quantized weights remain frozen.

**Hyperparameter details (Table 4).** The full training configuration includes:
- Optimizer: AdamW-8bit (8-bit quantized AdamW for additional memory savings)
- Policy learning rate: $1 \times 10^{-5}$ for QeRL and QLoRA, $5 \times 10^{-6}$ for BF16 LoRA (lower because BF16 LoRA is more fragile and collapses at higher learning rates, as shown in Figure 17)
- Training batch size: 128 prompts
- Samples per prompt ($G$): 8 for GSM8K, 16 for BigMath
- Policy updates per rollout ($\mu$): 4 for GSM8K (off-policy—reuses rollout data for multiple updates), 1 for BigMath (on-policy—discards data after one update)
- Max response length: 4096 tokens for GSM8K, 8192 for BigMath (longer chains of thought for harder problems)
- Rollout temperature: 1.0 (standard sampling, no temperature scaling that would affect entropy analysis)
- Clip range: $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$
- Noise range: $\sigma_{\text{start}} = 1 \times 10^{-2}$, $\sigma_{\text{end}} = 5 \times 10^{-4}$
- LoRA rank: 32 (default, with ablations at 16, 64, 128 in Figure 10)

**Design choice: learning rate asymmetry.** The paper explicitly notes that BF16 LoRA requires a lower learning rate ($5 \times 10^{-6}$) than QeRL and QLoRA ($1 \times 10^{-5}$) because "the fragile of the BF16 model with LoRA, the learning rate can not be larger than 5e-6, or it will collapse in the late training stage." This fragility is attributed to the full-precision model's lack of the stabilizing effect that quantization noise provides. In QeRL, the noise acts as an implicit regularizer—gradient updates that would cause large policy shifts in a deterministic 16-bit model are dampened by the stochastic perturbations, preventing the catastrophic collapse observed in the BF16 LoRA ablation at learning rate $3 \times 10^{-5}$ (Figure 17).

**On-policy vs. off-policy updates.** The paper uses different update strategies for different datasets: 4 off-policy updates per rollout for GSM8K (reusing the same generated outputs for multiple gradient steps), 1 on-policy update for BigMath (discarding rollouts after one gradient step). Off-policy updates are more sample-efficient (getting more learning per generated token), but they introduce bias because the policy has changed since the data was collected, making the importance sampling ratio $\pi_\theta / \pi_{\theta_{\text{old}}}$ deviate from 1. The paper compensates for this bias through the clipping mechanism in Equation 3, which prevents updates from becoming too large when the policy diverges from the behavior policy. The choice of 4 off-policy updates for GSM8K likely reflects the smaller dataset size (7,500 samples) and the need for data efficiency, while BigMath with its 122,000 samples can afford true on-policy updates.

**KL penalty handling.** For GRPO experiments (GSM8K), the KL penalty coefficient $\beta$ is included in the objective (Equation 3) but the paper states in Appendix E that it conducts experiments "without using entropy or KL losses"—this apparent contradiction may mean that $\beta$ is set to a very small nonzero value, or that the KL penalty is monitored but not backpropagated. For DAPO experiments (BigMath), the KL penalty is explicitly removed as per the DAPO algorithm design. This removal is particularly important for QeRL because AQN already adds entropy to the policy—adding a KL penalty on top would create a tension between the exploration encouraged by noise and the distributional conservatism enforced by the KL divergence, potentially negating QeRL's exploration benefit.

---

#### Summary of Design Choices and Their Justifications

- **NVFP4 over NF4:** NVFP4's dual-scaling floating-point format with block-wise FP8 scalers enables direct hardware acceleration via Marlin kernels, avoiding the lookup-table unpacking overhead that makes QLoRA's NF4 *slower* than BF16. The 16-element block size provides finer scaling granularity than MXFP4's 32-element blocks, which the paper shows leads to better final reward convergence (Figure 4).

- **LoRA on all linear projections:** covering $W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$ provides full expressivity for the trainable adapter while keeping only ~1% of total parameters trainable. This balances adaptation capacity against memory efficiency.

- **Exponential decay for noise schedule:** constant relative reduction per stage maintains meaningful exploration through middle training while rapidly dropping to near-zero noise in final stages, providing stable exploitation of learned strategies. Empirically superior to linear, cosine, and logarithmic schedules (Figure 9).

- **Noise merging into RMSNorm:** avoids breaking Marlin kernel compatibility (the kernel sees only clean NVFP4 × BF16 operations), requires zero additional parameters (noise is folded into existing RMSNorm scale vectors), and transforms additive noise into multiplicative noise (shown effective in RL literature). The per-block noise sharing (QKV group, gate-up group) further minimizes the number of independent noise sources.

- **Small $\sigma_{\text{start}} = 1 \times 10^{-2}$:** multiplicative noise in deep networks is more sensitive than additive noise used in traditional RL; the smaller initialization prevents destabilization of early LoRA training while still providing sufficient exploration pressure.

- **Learning rate asymmetry ($1 \times 10^{-5}$ for QeRL, $5 \times 10^{-6}$ for BF16 LoRA):** quantization noise acts as an implicit regularizer that stabilizes larger gradient updates, enabling QeRL to train with a 2× higher learning rate and achieve faster reward growth (Figure 16) without the catastrophic collapse observed in BF16 LoRA at $3 \times 10^{-5}$ (Figure 17).

- **Off-policy updates ($\mu = 4$) for GSM8K:** the smaller dataset size (7,500 samples) necessitates sample reuse to extract sufficient learning signal, with the PPO clipping mechanism compensating for the distribution shift between the old and current policies.

- **On-policy updates ($\mu = 1$) and higher clipping ($\epsilon_{\text{high}} = 0.28$) for BigMath:** the larger dataset (122,000 samples) enables true on-policy training, while the higher clipping bound and removed KL penalty (DAPO variants) maximize the model's ability to capitalize on the exploration driven by QeRL's AQN noise.

## 4. Key Insights and Innovations

### Innovation 1: Quantization Noise as an Intrinsic Exploration Mechanism — A Reframing of Quantization's Role in Training

The paper's most fundamental conceptual contribution is overturning the dominant framing of quantization in LLM training. Since the introduction of QLoRA (Dettmers et al., 2023a), the field has treated quantization as a **tolerated compromise** — a compression technique that trades precision for memory efficiency, where the goal is to minimize the degradation quantization causes. The entire research program around NF4, AWQ, GPTQ, and related methods centers on reducing quantization error to make the quantized model behave as closely as possible to the full-precision original.

QeRL shifts this framing from **"quantization as lossy compression"** to **"quantization as parameter-space exploration."** The key diagnostic move is the entropy analysis in Figure 5 and Figure 14, which shows that 4-bit quantized models consistently exhibit 0.07–0.10 nats higher sampling entropy than their 16-bit counterparts throughout RL training. This is not framed as a side effect to be minimized but as the **mechanism** by which quantization improves training — higher entropy means the model assigns meaningful probability to a wider range of tokens, leading to more diverse rollouts, which in turn increases the chance of discovering high-reward reasoning strategies that an overconfident full-precision model would never sample.

**What makes this distinctive:** the paper draws an explicit analogy to the parameter-space noise exploration literature in traditional RL (Plappert et al., 2017; Fortunato et al., 2018), where Gaussian noise is deliberately injected into network weights to drive exploration. The insight is that quantization error $\Delta\epsilon = Q(\theta) - \theta$ (Equation 5) is functionally identical to this injected noise — but emerges "for free" as a byproduct of compression rather than requiring explicit noise generation infrastructure. This reframing changes the evaluation criteria for quantization formats: instead of asking "which format minimizes output degradation?", one asks "which format produces the most beneficial exploration noise pattern?" This explains the paper's counterintuitive finding that quantization improves RL training (Table 1: NVFP4+LoRA achieves 88.5 on GSM8K for 7B vs. 88.1 for BF16 LoRA) despite degrading base model accuracy (73.4 vs. 76.3).

**Comparison to prior work:** The SFT paradigm (Dettmers et al., 2023a; Guo et al., 2023) universally treats quantization noise as harmful because SFT's objective is to faithfully imitate demonstration data. Noise in SFT corrupts the model's ability to reproduce exact reasoning traces. The paper explicitly contrasts this: "This finding contrasts with results from SFT of LLMs, demonstrating that controllable quantization noise in RL enhances exploration and enables quantized frameworks to surpass 16-bit LoRA in both efficiency and performance." This is not a small empirical tweak — it is a **domain-specific reframing** that says the value of quantization depends fundamentally on the training objective (imitation vs. exploration).

**Significance beyond performance:** This reframing opens a new axis for quantization research. If quantization noise is an exploration mechanism rather than just a precision loss, then the optimal quantization format for RL might differ from the optimal format for SFT or inference. The paper's comparison of NVFP4, MXFP4, and NF4 in Figure 4 provides initial evidence: all three 4-bit formats outperform 16-bit LoRA in reward growth, but NVFP4 and MXFP4 converge to higher final rewards than NF4, suggesting that floating-point quantization grids (with finer precision near zero) produce more structured exploration noise than NF4's data-dependent grid. This is a **fundamental conceptual advance** rather than an incremental improvement — it changes what researchers should optimize for when designing quantization schemes for RL training.

**Evidence anchor:** The entropy curves in Figure 5 (NVFP4-LoRA at ~0.22-0.28 nats vs. BF16-LoRA at ~0.15-0.18) and the consistent pattern across model sizes (Figure 14 for 14B) provide the empirical foundation. The reward curve comparisons in Figure 4 (all 4-bit formats outpacing BF16) show the downstream training benefit. The learning rate robustness experiment (Figure 16, where QeRL tolerates 3×10⁻⁵ while BF16 LoRA collapses) provides corroborating evidence that the noise stabilizes optimization.

---

### Innovation 2: Transforming Static Quantization Error into Dynamic Exploration via Adaptive Noise Scheduling

The paper identifies and solves a subtle but critical limitation of using quantization noise for exploration: quantization error is **static and deterministic** (fixed once weights are quantized), whereas effective RL exploration requires **dynamic and stochastic** perturbations whose magnitude can be scheduled to respect the exploration-exploitation trade-off. This is not merely an implementation detail — it is a **diagnostic insight** about why naïve quantization (without AQN) is insufficient despite providing an initial exploration benefit.

**What the field assumed before:** If quantization noise helps exploration (Innovation 1), one might conclude that simply quantizing the model and running RL is sufficient — the fixed quantization error pattern will provide a constant exploration bonus. This is implicitly what QLoRA-based RL approaches would do. The paper's AQN ablation (Figure 8) directly tests this assumption: training without AQN (static quantization noise only) shows early reward growth that plateaus and stagnates, while training with AQN continues to improve and achieves higher final rewards. The mechanism is that static noise explores a **single fixed perturbation direction** in parameter space — the model adapts to this particular perturbation pattern and converges to a local optimum defined by it. Dynamic noise (resampled per forward pass) explores **many perturbation directions**, preventing premature convergence and enabling the discovery of strategies inaccessible under the fixed perturbation.

**What makes this distinctive:** The innovation is not the noise injection itself (parameter noise is well-established in RL) nor the scheduling (noise decay is standard) but rather the **integration of these ideas into a quantization framework in a way that respects hardware constraints**. The Noise Merging technique (Equation 10, Appendix G) is the key conceptual bridge: by folding the injected noise into existing RMSNorm scaling parameters, QeRL achieves dynamic stochastic exploration with **zero additional memory or parameters** and without breaking Marlin kernel compatibility. This is a systems-level contribution as much as a conceptual one — it solves the practical problem of "how do you inject dynamic noise into a quantized model without dequantizing or adding overhead?" in a way that generalizes to any transformer architecture.

**Comparison to prior work:** FlashRL (Liu et al., 2025a) recognized that quantized rollouts save computation but introduced a precision-mismatch problem requiring importance sampling and dual-model memory. QeRL avoids this by keeping the same quantized model for rollouts and log-probability evaluation, with the noise injected transparently via RMSNorm so the model's output distribution remains self-consistent. Parameter-noise methods in traditional RL (Plappert et al., 2017) explicitly maintain and perturb noise parameters, adding memory and computation. QeRL's noise merging achieves equivalent functionality with zero overhead. The exponential decay schedule (Equation 8) is specifically motivated by the paper's empirical comparison of linear, exponential, cosine, and logarithmic schedules (Figure 9), showing that exponential decay's constant relative reduction per stage maintains meaningful exploration through middle training while enabling stable exploitation in late training.

**Why this is fundamental rather than incremental:** The transformation from static to dynamic noise is not just a refinement — it addresses the **inherent limitation** of using compression artifacts for exploration. Without it, quantization-enhanced RL would plateau at a ceiling determined by the fixed noise pattern. With AQN, the ceiling is lifted because exploration continues to sample diverse perturbation directions throughout training, with the schedule gradually shifting emphasis from exploration (high noise, early) to exploitation (low noise, late). The paper's evidence that AQN "effectively expands the model's exploration space, enabling further improvements in reward" near convergence (Section 4.2, Figure 8) suggests this is more than a minor tuning gain — it changes the asymptotic behavior of training.

**Evidence anchor:** Figure 8 (AQN vs. no-AQN reward curves for both 3B and 7B models) directly shows the dynamic-vs-static noise benefit. The noise scheduler comparison in Figure 9 isolates the exponential decay's contribution to late-stage stability. The learning rate robustness in Figure 16 (QeRL at 3×10⁻⁵ achieves nearly 2× faster reward growth than at 5×10⁻⁶ while remaining stable) is indirect evidence that AQN's dynamic noise regularizes the optimization landscape.

---

### Innovation 3: Empirical Demonstration That Quantization-Enhanced RL Systematically Outperforms Full-Precision PEFT on Reasoning Tasks

While Innovations 1 and 2 provide the conceptual framework, this innovation is the **empirical finding** that challenges a widely-held assumption in the LLM fine-tuning community: that quantization is a necessary evil that at best matches full-precision performance when done carefully. QeRL demonstrates that 4-bit quantized models with LoRA **consistently surpass** 16-bit LoRA on mathematical reasoning benchmarks, not just match them — and in some cases approach or exceed full-parameter fine-tuning. This finding spans multiple model sizes (3B, 7B, 14B, 32B), multiple datasets (GSM8K, MATH 500, AIME 24/25, AMC 23), and two RL algorithms (GRPO, DAPO), making it robust rather than anecdotal.

**The specific empirical patterns that make this a finding rather than a claim:**

1. **Quantization degrades base models, RL reverses this.** Table 1 shows NVFP4 quantization alone drops Qwen2.5-7B-Instruct GSM8K accuracy from 76.3 to 73.4 (a 2.9-point loss). But after RL training, NVFP4+LoRA achieves 88.5, surpassing BF16+LoRA at 88.1. The RL process doesn't just recover the quantization loss — it **overtakes** the full-precision baseline. This is not explainable by standard variance arguments; it requires a mechanism (entropy-driven exploration) that benefits the quantized model specifically.

2. **The advantage grows with model size and task difficulty.** For the 3B model on GSM8K (Table 1a), NVFP4+LoRA+AQN (83.7) substantially outperforms BF16+LoRA (76.1) — a 7.6-point gap. For the 7B model (Table 1b), the gap narrows to 1.7 points (90.8 vs. 88.1). For the 32B model on BigMath (Table 2), NVFP4+LoRA+AQN achieves 45.6 average across benchmarks vs. 42.2 for BF16+LoRA — a 3.4-point gap. The non-monotonic pattern suggests quantization's exploration benefit is most pronounced when the base model's capacity is just sufficient for the task (3B for GSM8K) or when the tasks are particularly challenging (32B on BigMath level 4-5).

3. **The benefit manifests in both training dynamics and final evaluation.** Training reward curves (Figures 4, 7, 12, 13) consistently show QeRL achieving faster reward growth — reaching comparable rewards in roughly half the steps of BF16 LoRA for the 7B model on BigMath (200 steps vs. 500+). This acceleration is practically significant: it means QeRL not only trains better models but trains them faster, compounding the per-step speedup from NVFP4 hardware acceleration.

4. **The effect generalizes across RL algorithms.** Figure 4 shows this pattern holds for both GRPO (top row) and DAPO (bottom row), suggesting the exploration benefit is not algorithm-specific but stems from the quantization noise's effect on the policy's sampling distribution — a more fundamental mechanism.

**Comparison to prior work:** The SFT literature (Dettmers et al., 2023a; Guo et al., 2023) treats quantization as a trade-off where performance loss is the cost of memory savings. QLoRA's central claim was that NF4+LoRA can *match* BF16 fine-tuning on SFT tasks — matching, not exceeding. The RL + PEFT work (Tina, Wang et al., 2025) showed that LoRA-based RL could work but didn't demonstrate quantization outperforming full precision. QeRL's finding that quantization *improves* over BF16 LoRA in RL while degrading it in SFT is not just a performance claim but a **domain-specific insight**: the value of quantization is training-objective-dependent.

**Significance beyond raw numbers:** This finding shifts the default for practitioners. If 4-bit quantized RL with LoRA systematically outperforms 16-bit LoRA while using ~40-50% of the GPU memory (Table 3) and running 1.2-1.5× faster, the rational choice for RL training on reasoning tasks becomes quantized by default, not by necessity. This inverts the standard hierarchy where full-precision training is the gold standard and quantization is a fallback for resource-constrained scenarios.

**Caveat the paper acknowledges:** The finding is demonstrated on mathematical reasoning benchmarks only (GSM8K, MATH 500, AIME, AMC). Whether the exploration benefit extends to other domains (code generation, general instruction following, creative tasks) where reward signals are less crisp or exploration requirements differ is unverified. The paper also does not test models above 32B parameters, leaving open whether the benefit scales to 70B+ models where quantization error patterns may differ.

**Evidence anchor:** Tables 1 and 2 provide the core quantitative evidence. Figures 4, 7, 12, and 13 show the training reward dynamics. The entropy analysis (Figures 5, 14) and learning rate ablation (Figures 16, 17) support the mechanistic explanation.

---

### Innovation 4: NVFP4 as a Quantization Format Specifically Suited to RL Training — Format Choice as an Architectural Decision

While NVFP4 was introduced as a hardware-supported 4-bit format for inference acceleration on Blackwell/Hopper GPUs, the paper demonstrates that its specific properties — dual-scaling with 16-element block-wise FP8 scalers, E2M1 floating-point grid, direct Marlin kernel support — make it **uniquely effective for RL training** in ways that are not reducible to throughput alone. This is a finding about **format-task alignment**: the same quantization format can serve efficiency and effectiveness goals simultaneously when those goals are properly understood.

**The format comparison that reveals this:** Figure 4 compares NVFP4, MXFP4, and NF4 under identical LoRA-based RL training. All three are 4-bit formats. All three provide similar memory reduction (~25-30% of BF16). Yet they produce different training outcomes:
- NF4: slowest reward growth and lowest final convergence, despite being the standard format for QLoRA-based fine-tuning.
- MXFP4: fast early reward growth (comparable to BF16 full-parameter training in early stages) but lower final convergence than NVFP4.
- NVFP4: best final convergence, matching or exceeding BF16 full-parameter training.

**What makes this distinctive:** If the only benefit of quantization were memory reduction, all three formats would produce similar training outcomes — they all compress to 4 bits. The divergence in reward curves indicates that **the structure of the quantization grid matters for exploration**. NVFP4's dual-scaling mechanism (per-tensor FP32 scaler + 16-element block-wise FP8 scalers) produces a quantization noise pattern that is:
- **Finer-grained** than MXFP4's 32-element blocks, giving it more degrees of freedom to capture local weight structure and produce exploration perturbations that preserve useful computations while introducing beneficial noise.
- **Floating-point-based** rather than data-dependent like NF4's Gaussian-quantile grid, producing noise correlated with weight magnitude (larger weights get proportionally larger perturbations) rather than with the global weight distribution statistics.

The floating-point grid also interacts more naturally with the additive Gaussian noise injected by AQN. Since FP4 quantization levels are geometric (spaced by powers of two), adding Gaussian noise in the RMSNorm scale space creates multiplicative perturbations whose effect scales with the quantized weight magnitude — a desirable property for exploration that NF4's irregular quantization grid does not provide.

**The throughput dimension:** NF4's poor RL performance is compounded by its hardware inefficiency — the lookup-table unpacking makes it 0.7-0.8× the speed of BF16 (Table 3), meaning NF4-based RL is both **slower** and **worse-performing** than BF16 LoRA. NVFP4's Marlin kernel support makes it 1.2-1.5× faster while also achieving better convergence. This **Pareto-dominant** outcome (better in both speed and quality) is what makes NVFP4 specifically compelling — it's not a trade-off between efficiency and effectiveness but a simultaneous improvement in both.

**Comparison to prior work:** Prior work on quantization formats (NF4 in Dettmers et al., 2023a; MXFP4 in Chmiel et al., 2025; NVFP4 in NVIDIA, 2024) focused on inference accuracy and throughput. The evaluation criterion was: how close is the quantized model's output to the full-precision model's output? QeRL demonstrates that for RL training, the relevant criterion is different: which format provides the most beneficial exploration noise pattern while maintaining hardware acceleration? This is not an incremental improvement in quantization quality — it's a **shift in the evaluation criterion** that could guide future format design for training workloads.

**Evidence anchor:** Figure 4 provides the direct three-format comparison under both GRPO and DAPO. Table 3 provides the hardware speedup comparison. The Marlin kernel throughput data in Tables 5-9 and Figure 11 quantify the practical advantage.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** Two primary mathematical reasoning datasets are used: **GSM8K** (Cobbe et al., 2021) comprising 7,500 training samples of grade-school math word problems, used for training 3B and 7B models with GRPO; and **BigMath** (Albalak et al., 2025) comprising 122,000 samples spanning difficulty levels 3–5, used for training 7B, 14B, and 32B models with DAPO. For the 32B model specifically, only the more challenging level 4–5 subset is used. Evaluation benchmarks include **GSM8K** (test set), **MATH 500** (Lightman et al., 2023), **AIME 2024**, **AIME 2025**, and **AMC 23** (Li et al., 2024), covering a range of difficulty from high-school competition math to olympiad-level problems.

- **Base model(s).** All experiments use the **Qwen2.5-Instruct** family (Team, 2024), specifically the 3B, 7B, 14B, and 32B parameter variants. The authors explicitly state they use "basic without any mathematic data fine-tuning" versions to evaluate training performance starting from general-purpose models rather than math-specialized ones. The models are quantized to NVFP4 using AWQ (Lin et al., 2024) with a calibration dataset of 256 sequences of 2048 tokens each, sampled from OpenThoughts-114k (Guha et al., 2025).

- **Metrics.** The primary metric throughout is **accuracy** (exact-answer match against ground truth), reported as **Pass@1** (average accuracy of a single sampled answer). For training monitoring, **accuracy reward** is tracked—the fraction of generated outputs within a training batch whose extracted final answer matches the ground truth, averaged across prompts. Entropy is measured as $H(\pi(\cdot|q)) = -\sum_{o_t \in V} \pi(o_t|q) \log \pi(o_t|q)$ over the vocabulary distribution at each decoding step. Training **throughput** is measured in tokens per second during the rollout phase, and **end-to-end training speedup** is reported as the wall-clock time per GRPO training step (rollout generation + log-probability computation + parameter updates).

- **Baselines.** The paper evaluates against five primary baselines:
  - **LoRA (BF16):** Standard 16-bit LoRA fine-tuning (Hu et al., 2022) with rank 32 on all linear projection matrices ($W_q, W_k, W_v, W_o, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). The base model weights remain in BF16.
  - **QLoRA (NF4):** QLoRA (Dettmers et al., 2023a) using 4-bit NormalFloat quantization of the base model with LoRA adapters. This represents the prior state-of-the-art for memory-efficient fine-tuning.
  - **Full-parameter training (BF16):** Full fine-tuning of all model parameters in BF16 precision—the gold standard for final performance.
  - **Base quantized models without RL:** NVFP4, MXFP4, and NF4 quantized models without any training, reported in Table 1 to quantify the degradation from quantization alone.
  - **NVFP4 + LoRA (no AQN):** QeRL without the Adaptive Quantization Noise mechanism—only the static quantization noise is present—used as an ablation baseline in Figure 8.

- **Generation budget / compute accounting.** The paper uses **total training steps** as the primary budget axis, with rewards tracked across steps (Figures 4, 7, 8, 12, 13). Training configurations differ by dataset: GSM8K uses a generation number of $G = 8$ outputs per prompt, while BigMath uses $G = 16$. For throughput and speedup measurements (Tables 3, 5–9, Figure 11), compute is measured in **tokens per second** during the rollout phase, with input length fixed at 256 tokens and maximum completion length at 2048 tokens. End-to-end speedup is measured as the inverse ratio of wall-clock time per complete GRPO training step, encompassing rollout generation, reward computation, log-probability evaluation, and gradient updates. The paper explicitly notes that speedup measurements for 3B and 7B models are conducted without gradient checkpointing or Liger loss to maximize training speed, while 14B and 32B models employ gradient checkpointing due to memory constraints.

- **Cross-validation / statistical protocol.** No formal cross-validation is described. The paper evaluates checkpoints at intervals between 500 and 1000 training steps. For evaluation on MATH 500, AIME, and AMC benchmarks, the paper reports that each dataset is "evaluated multiple times" using a temperature of 0.6, completion length of 4096, and top-p sampling with $p = 0.95$, with the average Pass@1 reported. No confidence intervals, error bars, or statistical significance tests are provided for any of the evaluation results, which is a notable methodological limitation.

---

### Main Quantitative Results

#### Reasoning Performance: QeRL Systematically Outperforms 16-Bit LoRA Across Model Sizes and Benchmarks

**GSM8K results (Table 1).** For the 3B model trained with GRPO, NVFP4+LoRA+AQN achieves **83.7%**, compared to 76.1% for BF16 LoRA (+7.6 points) and 84.4% for full-parameter BF16 training (−0.7 points). The base NVFP4-quantized 3B model (without any training) achieves only 59.4%, meaning the RL process adds 24.3 points of improvement from the quantized starting point. For the 7B model, NVFP4+LoRA+AQN achieves **90.8%**, surpassing BF16 LoRA at 88.1% (+2.7 points) and approaching full-parameter training at 91.2% (−0.4 points). Notably, the 7B model's base NVFP4 accuracy is 73.4%, so RL adds 17.4 points. Compared to QLoRA (NF4+LoRA), QeRL outperforms by 7.6 points (3B: 83.7 vs. 76.1) and 5.8 points (7B: 90.8 vs. 85.0).

A critical pattern emerges when comparing base model degradation to post-RL performance. Quantization alone degrades the 7B model by 2.9 points (76.3 → 73.4, Table 1b), but after RL training, the quantized model *overtakes* the BF16 LoRA model by 2.7 points (90.8 vs. 88.1). This reversal—quantization hurts base performance but helps RL-trained performance—is the core empirical evidence for the exploration-enhancement hypothesis.

**BigMath and multi-benchmark results (Table 2).** For the 7B model trained with DAPO on BigMath and evaluated across four benchmarks (MATH 500, AIME 24, AIME 25, AMC 23), QeRL (NVFP4+LoRA+AQN) achieves an average accuracy of **36.4%**, compared to 35.7% for BF16 LoRA (+0.7 points) and 37.3% for full-parameter training (−0.9 points). The base NVFP4 7B model averages 25.7% across these benchmarks—RL adds 10.7 points.

For the 14B model, QeRL achieves an average of **42.0%**, surpassing BF16 LoRA at 40.2% (+1.8 points) and approaching full-parameter training at 43.3% (−1.3 points). The 32B model shows the largest absolute gain: QeRL achieves **45.6%** average, outperforming BF16 LoRA at 42.2% (+3.4 points) and exceeding full-parameter training at 46.2% (−0.6 points). The 32B model on AMC 23 is particularly striking: QeRL achieves **63.3%**, substantially exceeding both BF16 LoRA (55.0%) and full-parameter training (57.5%), a gain of 8.3 and 5.8 points respectively.

**Pattern across model scales and difficulty.** The relative advantage of QeRL over BF16 LoRA appears largest for the 3B model (+7.6 points on GSM8K, Table 1a), moderate for the 7B model (+2.7 points on GSM8K, Table 1b; +0.7 points average on BigMath benchmarks, Table 2), and grows again for the 32B model (+3.4 points average, Table 2). This non-monotonic pattern suggests the exploration benefit from quantization noise is most impactful when the model's capacity is just sufficient for the task (3B for GSM8K) or when tasks are particularly challenging (32B on BigMath level 4–5), while being less pronounced for the intermediate 7B and 14B configurations where the base model already has adequate capacity.

**Comparison across quantization formats (Figure 4).** Under both GRPO and DAPO training, NVFP4 ultimately converges to the best final reward among 4-bit formats. MXFP4 shows faster early reward growth—in some cases matching BF16 full-parameter training in early stages—but converges to lower final rewards than NVFP4. NF4 consistently trails both NVFP4 and MXFP4 in reward growth and final convergence. This holds across both RL algorithms (GRPO top row, DAPO bottom row) and is consistent with the paper's claim that NVFP4's finer-grained block-wise scaling (16 elements vs. MXFP4's 32) and floating-point quantization grid provide more beneficial exploration noise patterns.

---

#### Reward Dynamics: QeRL Achieves Faster Convergence and Higher Final Rewards

**Training reward curves on BigMath (Figures 7, 12, 13).** For the 7B model on BigMath difficulty levels 3–5 (Figure 12), QeRL achieves a rapid reward increase after approximately **200 training steps**, reaching an accuracy reward of roughly 0.55. In contrast, BF16 LoRA requires over **500 steps** to achieve a similar reward level—more than 2.5× slower convergence. The final converged rewards show QeRL reaching approximately 0.55–0.60 while BF16 LoRA plateaus around 0.50–0.55. For the 14B model (Figure 7, left), QeRL shows a steep reward increase within the first 50–100 steps, significantly outpacing LoRA's more gradual climb. By step 250, QeRL reaches approximately 0.55 accuracy reward while LoRA is at roughly 0.40—a gap of about 0.15.

For the 32B model on the hardest data (levels 4–5, Figure 13), the difference in reward growth between QeRL and LoRA is less pronounced—both curves follow similar trajectories—but QeRL "still consistently performs better than LoRA," maintaining a small but persistent advantage through 250 steps. The paper attributes this smaller gap to the 32B model's already-strong base capabilities, which may reduce the marginal benefit of enhanced exploration.

**DAPO vs. GRPO comparison (Figure 4).** Under DAPO (bottom row), the reward advantage of quantized formats over BF16 LoRA appears more pronounced than under GRPO (top row). Specifically, NVFP4-LoRA under DAPO shows reward curves that closely track or even temporarily exceed BF16 full-parameter training, while under GRPO the quantized curves are more clearly below full-parameter. The paper does not explicitly analyze this difference, but it is consistent with DAPO's design choices (higher clipping upper bound, removed KL penalty) being synergistic with the exploration benefit from quantization noise—the policy is allowed to make larger updates toward high-reward strategies discovered through noise-driven exploration.

---

#### Memory and Throughput: QeRL Achieves 1.2–2.0× Speedup with 60–70% Memory Reduction

**Memory savings (Table 3).** For the 7B model, QeRL (NVFP4) reduces model memory from 15.2 GB (BF16 LoRA) to **5.9 GB**—a 61% reduction. QLoRA (NF4) achieves similar compression at 5.7 GB. For the 14B model, the reduction is from 29.6 GB to **10.6 GB** (64% reduction). These memory savings are what enable the headline capability: training a 32B model on a single H100 80GB GPU. The paper reports in Table 8 that the 32B model requires 62.3 GB in BF16 LoRA (exceeding the 80GB limit when accounting for activations, optimizer states, and multiple model copies), but only **20.7 GB** with NVFP4—comfortably fitting on a single GPU.

**End-to-end training speedup (Table 3).** With batch size 8, QeRL achieves **1.2× speedup** over BF16 LoRA for both the 7B and 14B models. QLoRA, in contrast, runs at **0.7×** the speed of BF16 LoRA—it is slower despite the memory reduction. This means QeRL is approximately **1.7–1.8× faster** than QLoRA end-to-end (1.2 / 0.7 ≈ 1.7). At batch size 2, the advantage grows: QeRL achieves 1.3× speedup for 7B and 1.3× for 14B, while QLoRA remains at 0.8–0.9×.

**Rollout phase speedup (Tables 5–8, Figure 11).** The speedup is driven primarily by the rollout generation phase. For the 14B model at batch size 2 (Table 7), QeRL achieves **95.3 tokens/s** vs. **65.4 tokens/s** for BF16 LoRA—a 1.46× speedup. For the 32B model at batch size 2 (Table 8), QeRL achieves **60.0 tokens/s** vs. **34.0**—a 1.76× speedup. The advantage scales with model size: the larger the model, the more the Marlin-accelerated NVFP4 kernel outperforms standard BF16 matrix multiplication. At batch size 8, the 32B model achieves 2.0× speedup (688.2 vs. 344.3 tokens/s).

**LoRA rank scaling (Table 9, Figure 11).** As LoRA rank increases, throughput decreases for both BF16 and NVFP4 due to the increased computational overhead of the adapter matrices. For the 14B model, BF16 throughput drops from 65.4 (rank 16) → 63.1 (rank 32) → 61.2 (rank 64), while NVFP4 drops from 95.3 → 92.9 → 86.0. Despite the decrease, NVFP4 maintains a consistent 1.4–1.5× speedup over BF16 at all ranks. For the 32B model, NVFP4 achieves 2.1–2.3× speedups across ranks—58.0 vs. 34.0 at rank 16, 56.0 vs. 33.3 at rank 32, 51.3 vs. 31.9 at rank 64. The paper notes these measurements use the vLLM engine with a batch size of 1 and equal GPU memory utilization settings between BF16 and NVFP4 variants.

---

#### Entropy Dynamics: Quantization Increases Sampling Entropy Throughout Training

**Entropy curves (Figures 5, 14).** Figure 5 shows the entropy trajectory for the Qwen2.5-7B-Instruct model during GRPO training. NVFP4-LoRA (rank 32) maintains entropy in the range of approximately **0.22–0.28 nats** across 200 training steps, while BF16-LoRA (rank 32) ranges from approximately **0.15–0.18 nats**—a consistent gap of about 0.07–0.10 nats. NF4-LoRA and MXFP4-LoRA both show intermediate entropy levels, starting around 0.18–0.20 and declining over training. Notably, while all models' entropy decreases as training progresses (reflecting the policy becoming more deterministic as it converges), the quantized models maintain a persistently higher entropy floor.

Figure 14 extends this to the 14B model on BigMath over 600 training steps. QeRL's entropy remains visibly above the BF16 LoRA baseline throughout, though both curves show a gradual downward trend. The paper highlights that the entropy difference is particularly pronounced in the initial training steps—exactly when exploration is most critical for discovering effective strategies. This aligns with the faster reward growth observed in Figures 7 and 12: the quantized models explore more broadly early on, discover high-reward strategies sooner, and then converge to those strategies as entropy naturally declines.

**Mechanism validation.** The entropy findings are not merely correlational—they provide the mechanistic link between the observation that quantization helps RL and the explanation (noise → entropy → exploration → faster reward growth). The consistent entropy gap across model sizes (7B in Figure 5, 14B in Figure 14), across quantization formats (all 4-bit formats show higher entropy than BF16), and across training steps (the gap persists even as both decline) strengthens the causal interpretation.

---

### Ablation Studies and Robustness Checks

**Adaptive Quantization Noise (AQN) vs. static quantization noise (Figure 8):** For both the 3B and 7B models, training with AQN (orange curve) consistently outperforms training without AQN (blue curve). In the 3B model, the two curves overlap during early training (steps 0–20), then diverge—the AQN curve continues to rise while the no-AQN curve plateaus around step 40–60 at approximately 0.65–0.70 accuracy reward, while AQN reaches roughly 0.80–0.85 by step 80. In the 7B model, the pattern is similar but less dramatic: AQN achieves approximately 0.55 at step 250 vs. roughly 0.50 without AQN—a smaller but consistent gap. The paper notes that AQN "effectively expands the model's exploration space, enabling further improvements in reward" near convergence, suggesting the dynamic noise allows the model to escape local optima that static quantization noise traps it in.

**Noise decay schedule (Figure 9, Figure 15):** Comparing exponential, linear, cosine, and logarithmic decay schedules for the 3B model, early training (steps 0–50) shows negligible differences among schedules. However, in later stages (steps 75–150), exponential decay achieves more stable improvement, reaching approximately 0.85 accuracy reward vs. roughly 0.75–0.80 for the other schedules. The paper attributes this to exponential decay maintaining smaller noise scales in later stages, enabling more stable exploitation. Figure 15 illustrates the actual noise curves: exponential decay drops sharply from $\sigma = 0.01$ in the first few stages (falling below 0.002 by stage 3), then gradually approaches $\sigma_{\text{end}} = 5 \times 10^{-4}$, spending most of training at low but non-zero noise levels.

**LoRA rank (Figure 10):** For the 3B model, ranks 16, 32, 64, and 128 exhibit "similar trends and reward growth rates." Rank 16 converges slightly faster in early training (reaching approximately 0.45 at step 20 vs. 0.35 for rank 128), and all ranks converge to similar final rewards around 0.70–0.75 by step 150. The paper concludes that rank 16 is "a more economical choice," suggesting the exploration benefit from quantization noise reduces the need for high adapter expressivity since the noise itself encourages the model to explore diverse parameter configurations implicitly.

**Learning rate (Figures 16, 17):** For the Qwen2.5-7B-Instruct model, QeRL trained with learning rate $3 \times 10^{-5}$ achieves roughly 0.95 accuracy reward by step 100 and converges by step 150, while QeRL at $5 \times 10^{-6}$ reaches approximately 0.90 by step 200. The higher learning rate thus achieves approximately **2× faster reward growth**. BF16 LoRA at $3 \times 10^{-5}$ collapses—the reward curve drops precipitously after step 50 (Figure 17)—while BF16 LoRA at $5 \times 10^{-6}$ trains stably but converges to lower final reward. The paper's interpretation is that NVFP4 quantization noise provides an implicit regularizing effect that stabilizes larger gradient updates, preventing the catastrophic policy collapse that full-precision LoRA experiences.

**Quantization format comparison within RL (Figure 4):** The three 4-bit formats show distinct reward trajectories. NVFP4 consistently achieves the best final convergence. MXFP4 often shows faster initial growth—in both GRPO and DAPO, MXFP4-LoRA matches or exceeds BF16 full-parameter training in the first 50–100 steps before plateauing. NF4 consistently underperforms both NVFP4 and MXFP4, with slower growth and lower final rewards. This three-way comparison controls for the quantization bit-width (all are 4-bit) and LoRA configuration (all use rank 32), isolating the effect of the quantization format's noise structure on RL training dynamics.

**Base model quantization degradation (Table 1):** Quantization alone degrades performance: NVFP4 7B drops from 76.3 to 73.4 (−2.9 points on GSM8K), MXFP4 7B drops to 71.3 (−5.0 points), NF4 7B drops to 70.5 (−5.8 points). The 3B model shows similar patterns: NVFP4 drops from 61.2 to 59.4 (−1.8), MXFP4 to 59.8 (−1.4), NF4 to 57.5 (−3.7). These degradations are smaller than the gains achieved through RL training (e.g., NVFP4 7B: 73.4 → 90.8, +17.4 points), confirming that RL training more than compensates for initial quantization loss.

---

### Critical Assessment

#### Claim 1: QeRL delivers over 1.5× speedup in the rollout phase.

The speedup data in Tables 5–8 and Figure 11 support this claim, but with important nuance about when and for which models. For the 14B model at batch size 2, the rollout speedup is 1.46× (95.3 vs. 65.4 tokens/s, Table 7). For the 32B model at batch size 2, it is 1.76× (60.0 vs. 34.0, Table 8), exceeding the 1.5× claim. However, for the 7B model at batch size 2, the speedup is 1.31× (151.6 vs. 115.4, Table 6), below 1.5×. At batch size 8, the 7B model achieves only 1.27× (2091.8 vs. 1641.1). For the 3B model (Table 5), the rollout speedup is essentially negligible at batch size 2 (157.0 vs. 151.2, 1.04×) and batch size 8 (2271.4 vs. 2226.3, 1.02×). The paper acknowledges that for 3B and 7B models, "we did not enable memory-efficient techniques such as gradient checkpointing or Liger loss in order to maximize training speed," suggesting the speedup is more modest for smaller models where the Marlin kernel's advantages are less pronounced relative to the baseline.

**Assessment:** The "over 1.5×" claim is supported for 14B and 32B models but overstated for smaller models. The practical significance depends on model scale—QeRL's acceleration is most impactful where it is most needed (large models that strain GPU memory), but the headline figure should be understood as model-size-dependent rather than universal.

#### Claim 2: QeRL enables RL training of a 32B LLM on a single H100 80GB GPU.

This claim is supported by the memory data. Table 8 shows the 32B model requires 62.3 GB in BF16 LoRA, which is functionally impossible on an 80GB GPU when accounting for activations, optimizer states, KV-cache for long sequences (up to 8192 tokens), and multiple model copies required by GRPO. QeRL reduces this to 20.7 GB. The paper reports that with QeRL, the 32B model trains at 10.6 seconds per step (rollout phase, without gradient checkpointing) and 12.2 seconds per step (with gradient checkpointing, batch size 8), confirming practical feasibility. However, the paper does not report whether this uses the full GRPO training loop (with reference model, multiple rollouts, etc.) or a simplified configuration. The memory numbers in the tables appear to be for model weights only, not peak training memory. Without reporting actual peak GPU memory utilization during a complete training step, the claim that the entire RL pipeline fits on a single GPU cannot be fully verified from the provided data.

**Assessment:** The model size reduction from 62.3 GB to 20.7 GB strongly suggests feasibility, but the paper should have reported peak training memory (including activations, optimizer states, and KV-cache) to make this claim fully verifiable. The practical value is high—enabling 32B RL training on commodity hardware—but the evidence is incomplete.

#### Claim 3: QeRL achieves faster reward growth than 16-bit LoRA and QLoRA.

The training reward curves (Figures 4, 7, 12, 13) consistently show quantized formats achieving steeper initial reward growth than BF16 LoRA. For the 7B model on BigMath (Figure 12), QeRL reaches ~0.55 reward at step 200 vs. BF16 LoRA requiring over 500 steps—approximately 2.5× faster convergence. For the 14B model (Figure 7), the gap is visibly large in the first 100 steps. However, the paper does not provide a quantitative metric for "faster reward growth" (e.g., area under the reward curve, steps to reach a threshold, or time-to-convergence), making the claim qualitative. The speedup numbers reported are for per-step throughput, not for convergence speed. A model that is 1.5× faster per step but requires the same number of steps to converge would have 1.5× faster time-to-convergence; a model that is 1.5× faster per step *and* requires half as many steps would have 3× faster time-to-convergence—the paper conflates these two effects without disentangling them.

**Assessment:** The visual evidence from the reward curves supports faster convergence, but the lack of a quantitative convergence-speed metric (e.g., "reaches 90% of final reward in X steps vs. Y steps for baseline") weakens the claim. The interaction between per-step throughput speedup and reduced steps-to-convergence is not systematically analyzed.

#### Claim 4: QeRL matches full-parameter fine-tuning on mathematical benchmarks.

For the 7B model, QeRL achieves 77.4 on MATH 500—matching full-parameter training exactly (Table 2, both at 77.4). On GSM8K, QeRL achieves 90.8 vs. full-parameter at 91.2 (−0.4 points, Table 1b). On the BigMath average across four benchmarks, the 7B QeRL achieves 36.4 vs. 37.3 for full-parameter (−0.9 points), the 14B achieves 42.0 vs. 43.3 (−1.3 points), and the 32B achieves 45.6 vs. 46.2 (−0.6 points). These gaps are small—less than 1 point for the 7B and 32B models—and likely within the range of run-to-run variance, though the paper provides no confidence intervals to assess this.

However, the "matching" claim has caveats:
- On AIME 24, the 7B QeRL (15.5) exceeds full-parameter (16.7) by +1.2 points, but on AMC 23, it trails by −2.5 points (42.5 vs. 45.0). The pattern is not uniform across benchmarks.
- The 32B model on AMC 23 is an outlier: QeRL achieves 63.3 vs. full-parameter at 57.5, exceeding by 5.8 points. This is the only case where QeRL substantially *outperforms* full-parameter training, and the paper does not explain this anomaly.
- The full-parameter baseline uses BF16 precision throughout—not the mixed-precision or quantization-aware training that might be used in practice for large-scale training.

**Assessment:** The claim of matching full-parameter performance is broadly supported (gaps of <1 point on average for 7B, 14B, and 32B models), but the paper should report variance across multiple training runs to distinguish genuine differences from noise. The anomalous AMC 23 result for the 32B model warrants investigation but is not analyzed.

#### Unaddressed Weaknesses

**No statistical rigor.** The paper reports point estimates for accuracy without confidence intervals, standard deviations, or any measure of uncertainty. Given that RL training is notoriously high-variance (the policy's performance can fluctuate significantly across seeds), and the evaluation uses temperature-based sampling (temperature 0.6, top-p 0.95) which introduces stochasticity, the reported numbers could easily vary by ±2–3 points across runs. No claims about "outperforming" or "matching" can be rigorously evaluated without variance estimates.

**Single model family.** All experiments use Qwen2.5-Instruct models. The exploration-enhancement hypothesis—that quantization noise increases entropy and benefits RL—depends on properties of the model architecture (how noise propagates through layers, the sensitivity of the output distribution to weight perturbations) that may differ across model families (e.g., LLaMA, DeepSeek, Mistral). Without experiments on at least one other model family, the generality of the findings is unverified.

**Missing combined baseline.** The paper never reports what happens when you apply AQN-style noise injection to a 16-bit BF16 LoRA model—only to quantized models. This would disentangle whether the benefit comes from noise injection (which could be applied to any model) or specifically from quantization noise interacting with AQN. If BF16+AQN outperforms BF16 LoRA, then the benefit is from noise injection, not quantization per se. If BF16+AQN does not help, then quantization's noise structure specifically enables the benefit. This ablation is essential for the paper's central claim that quantization noise (not just any noise) enhances RL.

**Incomplete memory reporting.** The paper reports model weight sizes (e.g., 5.9 GB for 7B NVFP4) but does not report peak training memory—including activations, optimizer states for LoRA parameters, KV-cache for generated sequences up to 8192 tokens, and the reference model copy required by GRPO. For the headline claim about 32B training on a single 80GB GPU, peak memory is the binding constraint, not static model size. The omission of peak memory numbers is a significant gap.

**No comparison to 8-bit quantization.** The paper compares against BF16 (16-bit), NF4 (4-bit), and other 4-bit formats, but not against 8-bit quantization (e.g., INT8 or FP8). An 8-bit baseline would help establish whether 4-bit specifically provides beneficial noise structure, or whether any reduced precision provides exploration benefits. If 8-bit LoRA also outperforms BF16 but is faster than 4-bit (since 8-bit can use native GPU tensor core operations), the optimal precision for RL training might be 8-bit, not 4-bit.

**Reward metric during training.** The paper uses accuracy reward (binary, based on exact-answer matching) to track training progress. This is a sparse and noisy metric, especially early in training when very few rollouts produce correct answers. The reward curves show substantial fluctuation (visible in Figures 7, 8, 12), and the paper does not apply any smoothing or report confidence bands. Early-training comparisons based on noisy reward curves are particularly unreliable.

**Temperature and entropy.** The paper reports that quantization increases sampling entropy at temperature 1.0 (the rollout temperature used during training). However, the relationship between weight perturbation and sampling entropy is temperature-dependent: at lower temperatures, the softmax sharpens, potentially suppressing the entropy-increasing effect of quantization noise. The paper does not investigate whether the entropy benefit persists at different temperatures, which matters because practitioners often tune sampling temperature during RL.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Cost Is Unaccounted For — Making the Headline Efficiency Gains an Upper Bound

**The assumption or constraint.** The compute-optimal framework rests entirely on the ability to estimate prompt difficulty before allocating the inference budget. The paper's method for estimating difficulty — generating 2048 samples per question and averaging either ground-truth correctness (oracle) or PRM final-answer scores (predicted) — is extraordinarily expensive. The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This means the difficulty estimation step alone consumes more compute than the largest test-time budgets studied (256–512 generations), but this cost is excluded from all efficiency calculations in the paper.

**The consequence.** The headline claim of a `4×` improvement in compute efficiency over best-of-N (Figures 4 and 8) is computed *after* difficulty is already known, without amortizing the cost of learning it. In a realistic deployment where difficulty must be estimated per-query, the total cost would be difficulty estimation plus strategy execution, and the former could dominate the latter. A practitioner deploying this system would find that the `4×` figure represents an idealized upper bound that is unattainable in practice unless a much cheaper difficulty estimation method is developed. The true efficiency gain — accounting for the amortized cost of 2048 samples per query — could be substantially smaller, possibly even negative for low-volume deployment scenarios where the estimation cost cannot be amortized across many queries.

**What evidence exists in the paper.** The paper provides no experiment or analysis that accounts for difficulty estimation cost in any budget calculation. The cross-validation protocol (Section 3.2) does select strategies per difficulty bin, but the cost of generating the 2048 samples to assign each test question to a bin is never included in the compute budgets reported in Figures 4 and 8. The overlap of oracle and predicted difficulty curves in these figures demonstrates that the PRM-based estimation works without ground-truth labels, but says nothing about its cost.

**Mitigation status.** The paper acknowledges this gap explicitly and flags it as "a key avenue for future work" (Section 3.2), suggesting that future systems could train models to predict difficulty directly from question text or use adaptive estimation that amortizes difficulty assessment into the solution process. No such model or method is developed or evaluated in this work. Until this gap is closed, the `4×` figure should be understood as an **upper bound on achievable efficiency** rather than a realized deployment gain, and any practitioner considering this approach must budget for the difficulty estimation overhead separately.

---

### Hard Problems Remain Essentially Unsolved — Test-Time Compute Cannot Create Capability From Nothing

**The assumption or constraint.** The entire compute-optimal framework operates on the premise that the base model already produces correct solutions at some non-trivial rate — the pass@1 on a given problem must be measurably above zero for test-time compute to amplify it. For the hardest problems (difficulty bin 5), this premise fails. The paper is explicit about this boundary:

> "on the hardest questions (bin 5), no method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of how the budget is allocated" (Section 5.3)

**The consequence.** This is not merely a quantitative shortfall but a fundamental capability ceiling. Across all methods — search, revisions, and their compute-optimal combinations — bin 5 accuracy hovers at 1–3% regardless of compute budget. In Figure 3 (right), bin 5 shows near-zero accuracy for both beam search and best-of-N at all budget levels. In Figure 7 (right), bin 5 shows roughly 2–3% accuracy irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling line is essentially flat near 0–5%. The consequence for practitioners is clear: if your problem distribution contains a substantial fraction of genuinely hard problems (outside the base model's capability range), no amount of test-time compute — regardless of how cleverly allocated — will help. These problems require either a more capable base model (via scaling pretraining) or a fundamentally different approach. The paper's framework provides no mechanism for bridging capability gaps, only for amplifying existing capabilities.

**What evidence exists in the paper.** The evidence for this limitation is pervasive and consistent across every experiment that breaks out results by difficulty bin. Figure 3 (right, bin 5), Figure 7 (right, bin 5), Figure 9 (bin 5), and the FLOPs-matched bar charts in Figure 1 (showing -37.2% to -52.9% relative disadvantage for hard problems) all converge on the same finding. The paper deserves credit for being transparent about this boundary rather than overclaiming.

**Mitigation status.** The paper does not attempt to solve this limitation. Section 7 explicitly concludes that "test-time compute can amplify existing capability but does not create it from nothing." The takeaway box in Section 8 reinforces this: for hard problems, pretraining remains the only viable path. The paper does not suggest any mechanism by which test-time compute could be extended to address capability gaps, nor does it explore whether the difficulty threshold could be pushed lower through better verifier training or more sophisticated exploration strategies. This is an honest acknowledgment of a hard boundary rather than a solvable limitation — but it means the framework's applicability is strictly limited to problems within the base model's approximate capability range.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate — And Training It Is Fragile

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect followed by a correct target. At test time, the model may encounter correct answers in its context (produced during earlier revision steps) and — having never seen this pattern during training — will often incorrectly "revise" a correct answer into a wrong one. The paper reports:

> "approximately 38% of correct answers produced during a revision chain get 'revised' back to incorrect answers in the subsequent step" (Section 6.1)

**The consequence.** This reversion rate means that longer revision chains are not monotonically beneficial — adding more revision steps eventually becomes counterproductive as the model cycles between correct and incorrect answers. The paper's mitigation is to use majority voting or verifier-based selection across the entire chain rather than taking the final output, but this is a post-hoc heuristic, not a solution to the underlying training distribution mismatch. In practice, this means the revision model cannot be trusted to converge to a correct answer through iterative refinement alone — an external selection mechanism is always required. Furthermore, the ReST^EM experiment (Appendix K, Figure 16) reveals that the revision training procedure is fragile: attempting to further optimize the revision model through on-policy RL training caused performance to degrade substantially with sequential revisions — performance dropped to approximately 33.5% at 256 fully-sequential generations compared to roughly 38.5% at the optimal ratio. This suggests that the positive revision results depend on specific offline data construction choices (edit-distance-based incorrect-correct pairing) that may not transfer to other settings or survive further optimization.

**What evidence exists in the paper.** The 38% reversion rate is reported in Section 6.1 based on empirical measurement. The ReST^EM failure is documented in Appendix K and Figure 16, with the authors hypothesizing that "on-policy data collection in ReST^EM exacerbates spurious correlations in revision data, causing the model to fail to learn the revision task properly." Importantly, the paper does not provide a breakdown of when reversion occurs — e.g., whether it happens more on easy problems (where the initial answer is likely correct) or hard problems (where initial answers are likely wrong and the model has never seen a correct-to-correct transition).

**Mitigation status.** The paper partially mitigates this through within-chain selection (majority voting or verifier) that can pick the best answer from any point in the revision chain. This works in practice but does not address the root cause — the training data distribution mismatch. A more principled solution, such as training the model on trajectories that include correct-to-correct transitions or teaching the model to recognize when no revision is needed, is not explored. The ReST^EM negative result suggests that naively applying standard RL optimization to revision models can make things worse, creating a barrier to further improvement through self-play or iterative training. A practitioner deploying the revision approach should expect to implement external selection and should not rely on the final revision output being the best.

---

### Sequential Revision Strategies Create a Latency Penalty That the Paper Does Not Address

**The assumption or constraint.** The paper measures compute in "generations" (number of complete solutions sampled), which is a reasonable proxy for total FLOPs but ignores wall-clock latency. Sequential revisions are inherently serial — each revision depends on the previous one and cannot be parallelized. A strategy that allocates 128 generations as 64 sequential × 2 parallel takes roughly 64× longer in wall-clock time than one that runs 128 parallel samples simultaneously with sufficient hardware. The paper's compute-optimal policy frequently favors sequential-heavy strategies, particularly on easy problems where "fully sequential revisions dominate" (Section 6.1) and on medium problems where a balanced ratio is optimal (Figure 7, right).

**The consequence.** For latency-sensitive applications — interactive assistants, real-time tutoring systems, live code generation — the sequential-heavy strategies favored by the compute-optimal policy may be completely impractical regardless of their accuracy advantages. A user waiting for an answer from a math tutoring system would experience a sequential revision chain of 64 steps as an unacceptably long delay, even if the total FLOPs consumed is equivalent to a parallel strategy that returns results in 1/64th the time. The paper's analysis treats all "generations" as having equal cost, but in deployment, latency and throughput are distinct constraints, and the sequential strategies that maximize FLOPs-efficiency may minimize latency-efficiency. This is particularly acute for the revision model, where the entire chain of sequential revisions must complete before the final answer selection can occur (since within-chain selection examines all steps).

**What evidence exists in the paper.** The paper provides no latency analysis whatsoever. No wall-clock time measurements are reported for revision chains or beam search steps. The FLOPs-matched comparison (Section 7) uses total generation count as the cost metric, which implicitly assumes all generations can be executed in parallel if desired — an assumption that breaks for sequential dependencies. The speedup measurements in Section 4.3 are for per-step training throughput, not for inference latency in deployment. The paper does not discuss the latency-throughput tradeoff or acknowledge that sequential and parallel strategies have different latency profiles even at equal FLOPs.

**Mitigation status.** This limitation is not acknowledged or addressed in the paper. The compute-optimal policy selects the best strategy per difficulty bin purely based on accuracy at a given generation budget, with no latency constraint in the optimization objective (Equation 1). A latency-aware formulation would add a constraint on maximum sequential depth or would penalize sequential strategies in the utility function. Future work could extend the framework to optimize a combined accuracy-latency objective, but in its current form, the compute-optimal policy may produce recommendations that are latency-infeasible for real-time applications. Practitioners deploying this system should augment the policy selection with a latency budget and possibly restrict the sequential-to-parallel ratio based on their specific latency requirements.

---

### All Results Are on a Single Benchmark (MATH) with a Single Model Family (PaLM 2-S\*) — Generality Is Unproven

**The assumption or constraint.** Every experiment in the paper uses the MATH benchmark (500 test questions) with PaLM 2-S\* (Codey) as the base model. The authors state they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified. The paper provides no experiments on other reasoning benchmarks (e.g., GSM8K, MMLU, ARC), no experiments on other model families (e.g., LLaMA, Mistral, DeepSeek), and no experiments on non-math reasoning tasks (code generation, logical reasoning, scientific QA). The difficulty estimation mechanism, the PRM's behavior, and the revision model's effectiveness all depend on properties of the model's output distribution — calibration, error patterns, in-context learning ability — that vary substantially across model families and tasks.

**The consequence.** Several aspects of the paper's findings could be model-specific or benchmark-specific:
- The PRM's over-optimization behavior (beam search degrading on easy problems) depends on how reliably the PRM scores PaLM 2-S\*'s outputs. A model with different error patterns or calibration might show different difficulty-dependent scaling curves, potentially changing which strategies are optimal at which difficulty levels.
- The revision model's ability to learn from incorrect in-context examples depends on PaLM 2-S\*'s in-context learning capabilities and the specific edit-distance-based data construction. Other model families may have different in-context learning strengths, potentially making the revision approach more or less effective.
- MATH consists of competition-level math problems requiring symbolic manipulation and multi-step deduction. The finding that revisions help on easy problems and search helps on medium problems might not generalize to domains requiring factual recall (where the failure mode is lack of knowledge, not incorrect reasoning) or to open-ended generation (where correctness is ambiguous).

**What evidence exists in the paper.** Zero cross-benchmark or cross-model experiments are provided. The paper reports only MATH results with PaLM 2-S\*. The difficulty bins are computed relative to PaLM 2-S\*'s pass@1 rates and may not correspond to difficulty bins for other models — the same MATH problem could be in bin 2 for one model and bin 4 for another, and the optimal strategy could differ accordingly. The paper acknowledges this limitation implicitly by noting that the difficulty bins are model-specific (Section 3.2), but does not test whether the overall framework (difficulty-conditioned allocation) transfers.

**Mitigation status.** The paper does not address this limitation beyond stating a belief in the model's representativeness. The authors do not claim generality beyond MATH and PaLM 2-S\*, but the paper's framing as a general framework for compute-optimal test-time scaling implies broader applicability that remains unproven. Replication on at least one other benchmark and one other model family would substantially strengthen the claims. In the current state, a practitioner using a different model (e.g., LLaMA 3, GPT-4) on a different task (e.g., code generation) should expect to re-derive the compute-optimal policies from scratch rather than applying the specific strategies (beam search on medium, sequential on easy) reported in this paper, since the difficulty-dependent patterns may differ.

---

### The 14× Larger Model Baseline Is Weakened by Non-Compute-Optimal Pretraining and No Test-Time Compute

**The assumption or constraint.** The FLOPs-matched comparison in Section 7 compares PaLM 2-S\* with compute-optimal test-time scaling against a model with approximately `14×` more parameters, trained on the same data (parameter-only scaling, following the LLaMA paradigm rather than Chinchilla-optimal joint scaling), and using only greedy decoding with no test-time augmentation. The authors acknowledge this departure from compute-optimal pretraining:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work." (Section 7)

**The consequence.** The `14×` larger baseline is weaker than it needs to be in two ways:
- A Chinchilla-optimal model trained with `14×` more total FLOPs (scaling both parameters and data) would likely outperform a parameter-only-scaled model, potentially erasing or reversing the advantages reported for test-time compute. The paper's Figure 1 bar charts showing +27.8% relative improvement for test-time compute on easy questions at `R ≪ 1` might shrink or become negative against a properly compute-optimal larger model.
- The larger model uses only greedy decoding — no majority voting, no best-of-N, no search, no revisions. This is an asymmetric comparison: the smaller model gets sophisticated test-time strategies while the larger model gets none. A fairer comparison would give the larger model some test-time compute budget proportional to its per-token cost — even a modest best-of-8 would substantially improve the larger model's performance and likely change the FLOPs-matched outcome. The paper's finding that "test-time compute can outperform a `14×` larger model" should be understood as "test-time compute can outperform a `14×` larger *parameter-only-scaled* model *using greedy decoding*" — a much narrower claim.

**What evidence exists in the paper.** The paper explicitly describes the baseline: scaling parameters only while fixing data, greedy decoding. No ablation is provided where the larger model also receives test-time compute. The Chinchilla-optimal comparison is deferred to future work. The paper does not report how much the larger model would improve with even a small test-time compute budget (e.g., best-of-4 or best-of-8), which would help practitioners assess whether the asymmetry in the comparison drives the result.

**Mitigation status.** The paper is transparent about this limitation, acknowledging both the parameter-only scaling and the lack of test-time compute for the larger model. However, the transparency does not fix the weakened baseline. The practical implication is that the FLOPs-matched results represent an **upper bound on the advantage of test-time compute** — against a stronger baseline (Chinchilla-optimal training plus modest test-time compute), the advantage would narrow or potentially disappear for all but the easiest problems. A practitioner deciding between training a larger model and investing in test-time compute infrastructure should treat the paper's FLOPs-matched results as suggestive but not definitive, and would ideally run their own comparison with a truly matched baseline (compute-optimal training for the larger model, proportional test-time compute budget for both).

## 7. Implications and Future Directions
- Shift in how we view quantization in RL
  - By demonstrating that quantization noise can serve as a built‑in exploration mechanism, QeRL bridges compression and RL theory. This invites systematic study of “noise‑aware” RL schedules and quantization formats tailored for exploration (Sec.3.2–3.3; Fig.3, Fig.5).

- Practical impact: lower barrier to large‑model RL
  - Memory and speed gains make RL feasible for larger models and on fewer GPUs. The single‑GPU 32B result suggests RL fine‑tuning could become routine in more labs and production teams (Abstract; Tables 7–8).

- Extensions and research directions
  - Beyond math: test on code generation, tool use, and general instruction‑following to assess how quantization‑driven exploration interacts with different reward structures.
  - Broader noise design: explore alternative noise distributions, layer‑wise schedules, or adaptive controllers that react to online reward/entropy signals rather than pre‑set schedules (Fig.9 hints scheduler choice matters).
  - Combine with activation quantization/QAT: integrate low‑bit activations or quantization‑aware training to further reduce memory and possibly shape exploration at activation level.
  - Algorithmic integration: unify AQN with explicit entropy bonuses or KL constraints (e.g., marrying AQN with DAPO’s no‑KL regime vs GRPO’s KL penalty; Sec.3.1).
  - Scaling studies: push beyond 32B to 70B+ to test whether QeRL’s speed/memory gains and exploration benefits persist.

Key takeaway
> QeRL reframes quantization from a necessary evil into a controllable exploration tool for RL, and couples it with a hardware‑aligned 4‑bit format (NVFP4) and zero‑overhead noise injection. The result is both faster and often more accurate RL training than 16‑bit LoRA and QLoRA, with evidence across 3B–32B models and multiple math benchmarks (Fig.1; Tables 1–3).
