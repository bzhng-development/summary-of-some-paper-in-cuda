# A Survey of Small Language Models

**ArXiv:** [2410.20011](https://arxiv.org/abs/2410.20011)

## 🎯 Pitch

This paper delivers the first comprehensive survey focused on Small Language Models (SLMs), introducing a novel taxonomy that links concrete architectural, training, and compression techniques to the specific computational, memory, and latency constraints they solve. By systematically organizing and synthesizing the literature across lightweight architectures, efficient attention, and resource-aware model optimization, it equips researchers and practitioners with an essential roadmap for deploying powerful language models in constrained environments—unlocking real-time, on-device, and privacy-preserving applications that traditional LLMs struggle to support.

---

## 1. Executive Summary (2–3 sentences)
This survey systematizes the fast-growing area of Small Language Models (`SLMs`) by introducing a clear taxonomy that maps concrete techniques (architectural design, training, and compression) to the specific constraints they alleviate (compute, memory, latency, storage, etc.). It then synthesizes the core mechanisms behind lightweight architectures, efficient attention, neural architecture search, pretraining/finetuning under resource constraints, and compression (pruning, quantization, distillation), along with evaluation settings, metrics, applications, and open problems (hallucination, bias, energy, privacy).

## 2. Context and Motivation
- Problem and gap addressed:
  - Large Language Models (`LLMs`) perform strongly but are expensive to train and run, typically requiring specialized hardware and centralized deployment (Introduction).
  - There is increasing demand for models that can run under constraints of training/inference hardware, data availability, bandwidth, or latency—on-device, mobile, and edge settings in particular (Abstract; Section 1).
  - The term “small” is relative and shifts over time and context; the field lacks a structured way to reason about “smallness” beyond raw parameter count (Section 1).

- Why this matters:
  - Practical: Enabling privacy-preserving, low-latency, and cost-efficient deployment on consumer devices and edge infrastructure can unlock real-time assistants, offline capabilities, and compliance-sensitive domains (healthcare, finance) (Abstract; Sections 6, 7).
  - Scientific/engineering: Understanding the trade-offs between memory, compute, and performance guides algorithmic choices and hardware–software co-design, improving efficiency without sacrificing capability (Table 1; Section 2–4).

- Prior approaches and limitations:
  - Existing surveys focus on general LLMs or specific learning methods, not SLMs as a category (Section 1).
  - Individual works address isolated bottlenecks (e.g., attention complexity, compression) but practitioners lack a unifying map linking methods to constraints and applications.

- Positioning of this survey:
  - It proposes a “two-axis” taxonomy to organize methods by:
    - Where they act: model architecture, training, or post-training compression.
    - Which constraint they optimize (e.g., training compute, inference runtime, memory, storage, latency).
  - As the authors summarize:
    > “We propose a novel taxonomy for organizing the methods along two axes: • the techniques used in pre-processing (model architecture), training, and post-processing (model compression) SLMs; and • the constraints the technique is attempting to optimize for...” (Section 1; Table 1, Table 2)
  - The survey claims novelty as the first focused synthesis of SLMs (Section 1).

## 3. Technical Approach
This survey’s “methodology” is a structured organization and deep explanation of how SLM-enabling techniques work. It groups techniques into: (i) architectural design, (ii) training under constraints, and (iii) post-training compression, then cross-references each to the constraints they primarily relieve (Table 1). Below, mechanisms are explained in plain terms with targeted definitions of uncommon concepts.

- The taxonomy (how to use it)
  - Table 1 matches techniques to constraints (training compute, dataset size, inference runtime, memory, storage, latency). For instance:
    - Pruning and quantization directly affect inference-time memory, storage, and latency.
    - Efficient attention and lightweight architectures reduce inference runtime and memory.
    - Mixed-precision pretraining and distributed sharding reduce training compute and memory.
  - Table 2 then links real-world settings (e.g., on-device, privacy-preserving, efficient inference, energy-efficient AI) to datasets and metrics appropriate for measuring improvements in those constraints.

- Section 2: Model architectures
  - 2.1 Lightweight architectures (encoder-only and decoder-only)
    - Idea: Redesign modules to deliver similar quality with fewer parameters or cheaper operations.
    - Examples and mechanisms:
      - `MobileBERT`: introduces an inverted-bottleneck structure (a narrow–wide–narrow block) to balance attention and feed-forward costs, yielding substantial efficiency gains:
        > “...achieving a 4.3x size reduction and a 5.5x speedup compared to the base version of BERT.” (Section 2.1)
      - `DistilBERT`, `TinyBERT`: knowledge distillation to reduce size while retaining accuracy (Section 2.1).
      - `TinyLLaMA` (≈1.1B): uses memory-optimized kernels like `FlashAttention` (an IO-aware exact attention algorithm that lowers memory via tiled, streaming computation) to keep performance competitive at small sizes (Section 2.1).
      - `MobileLLM`, `MobilLLaMA`: parameter sharing (reusing blocks), grouped-query attention, and embedding sharing to reduce latency and memory (Section 2.1).

  - 2.2 Efficient self-attention approximations
    - Challenge: Standard attention scales quadratically in sequence length N (O(N^2)).
    - Mechanisms:
      - `Reformer` uses locality-sensitive hashing (`LSH`) to approximate nearest neighbors in attention, reducing complexity to O(N log N) (Section 2.2).
      - Linear-attention variants approximate attention as kernel feature map products, achieving O(N) time and space; one can view the model as a recurrent computation for faster inference (Section 2.2).
      - State-space sequence models (`Mamba`) and hybrid RNN-like designs (`RWKV`) deliver linear-time/space sequence modeling by maintaining a state that evolves with inputs, avoiding quadratic attention altogether (Section 2.2).
      - Nyström-based approximations select landmark points to approximate the full attention matrix efficiently (Section 2.2).
    - Why these over alternatives: They directly target the quadratic bottleneck and enable longer contexts or faster throughput within tight memory budgets.

  - 2.3 Neural Architecture Search (`NAS`)
    - Problem: Exhaustive search is prohibitively expensive for billion-scale LMs.
    - Strategies summarized:
      - Constrain search over depth/width (e.g., number of layers/heads) in sub-billion regimes (e.g., `MobileLLM`) to find sweet spots (Section 2.3).
      - Reduce search space and initialize well to speed convergence (Section 2.3).

  - 2.4 Small multi-modal models
    - Mechanisms to reduce total parameters while aligning vision and language:
      - Use smaller LMs (e.g., `Gemma`, `phi-3-mini`) and carefully curated multimodal data (Section 2.4).
      - Shrink or prune vision encoders (e.g., leverage intermediate layers, discard later blocks) or replace them entirely with learned tokenizers:
        > “Monolithic multimodal models... eliminate the visual encoder, instead using lightweight architectures to generate visual tokens.” (Section 2.4)
      - Examples: `Chameleon` uses `VQ-VAE` (a vector-quantized autoencoder that maps images to discrete tokens), `Mono-InternVL` uses an MLP to generate visual tokens and a modality-specific mixture-of-experts feed-forward (Section 2.4).

- Section 3: Training techniques under constraints
  - 3.1 Pre-training
    - Mixed precision training: Use low-precision arithmetic (e.g., `FP16`, `BFLOAT16`) for forward/backward passes while maintaining a high-precision master copy for stability. `BFLOAT16` preserves a wider exponent range vs. FP16, improving stability (Section 3.1).
    - Emerging hardware formats:
      > “NVIDIA’s latest Hopper architecture introduces support for 8-bit floating-point (FP8) precision... enabling even greater computational efficiency...” (Section 3.1)
    - Optimizer and stability choices: Memory-efficient `Adafactor`, second-order-inspired `Sophia`, gradient clipping, careful initialization (Section 3.1).
    - Distributed sharding to fit larger batches/models:
      - `ZeRO-1/2/3` partitions optimizer states → gradients → parameters across devices; `FSDP` is a similar fully sharded approach (Section 3.1).

  - 3.2 Fine-tuning
    - Parameter-Efficient Fine-Tuning (`PEFT`):
      - `LoRA`: inserts low-rank adapters per weight matrix and only trains these small “delta” matrices (Section 3.2.1).
      - Prompt tuning and attention-block adapters (e.g., `Llama-Adapter`), plus dynamic adapter routing (mixtures of adapters) for multitasking while reducing forgetting (Section 3.2.1).
    - Data augmentation for instruction tuning:
      - Generate paraphrases or more complex instructions (`AugGPT`, `Evol-Instruct`), refine instruction–response pairs (`Reflection-tuning`), enrich with retrieval (`FANNO`), and synthesize hard examples iteratively (`LLM2LLM`) (Section 3.2.2).
      - Especially useful in low-resource, clinical, and privacy-sensitive domains (Section 3.2.2).

- Section 4: Model compression (post-training)
  - 4.1 Pruning
    - Unstructured pruning (remove individual weights):
      - `SparseGPT`: formulate pruning as layer-wise sparse regression to minimize output error efficiently even at massive scale (Section 4.1).
      - `Wanda`: prune using both weight magnitude and activation statistics, avoiding retraining weight updates:
        > “...incorporates both weights and activations... and eliminates the need of weight updates.” (Section 4.1; Appendix A)
      - `n:m pruning`: exactly n of every m weights are pruned (a semi-structured pattern) to unlock hardware speedups (e.g., TensorRT on GPUs) (Section 4.1; Appendix A).
    - Structured pruning (remove channels/heads/layers):
      - Exploit neuron/activation sparsity and architectural redundancy (e.g., prune attention heads, layers, or switch activations to ReLU to induce sparsity, then fine-tune) (Section 4.1).
      - Input-dependent “contextual sparsity” and dynamic KV-cache pruning for speed and memory gains (Section 4.1).

  - 4.2 Quantization
    - Weight-only vs. weight+activation:
      - `GPTQ`: layer-wise weight-only quantization minimizing reconstruction error using Hessian approximations (Section 4.2).
      - `AWQ`, `ZeroQuant`: use activations to assess weight importance and improve calibration (Section 4.2).
    - Activation quantization challenges:
      - Outliers (rare, large values) cause large errors when quantized. Remedies:
        - `SmoothQuant`: balance the dynamic range by shifting scale from activations to weights (“migrating quantization difficulty”) (Section 4.2).
        - `SpinQuant`: learn rotations to transform outliers into a more quantization-friendly space (Section 4.2).
    - K/V cache quantization:
      - The `KV cache` stores past attention keys/values in autoregressive decoding to avoid recomputation; quantizing it is critical for long contexts (Section 4.2).
    - Quantization-aware training (`QAT`):
      - Train the model with simulated quantization in the loop; can use distillation from a float16 teacher to recover accuracy (`LLM-QAT`, `EdgeQAT`) (Section 4.2).

  - 4.3 Knowledge Distillation (`KD`)
    - Distill “teacher” model behavior into a smaller “student” model via loss functions over logits/sequences.
    - Extensions and practicalities:
      - Very small students can outperform pretraining when distilled from strong teachers (e.g., 58M `BabyLLaMA`) (Section 4.3).
      - Sequence-level divergences (e.g., generalized f-divergences), calibration-aware losses, and fusion of multiple teachers (merge probability distributions) (Section 4.3).
      - Tokenizer mismatch and lack of teacher pretraining data complicate KD; universal logit distillation inspired by optimal transport helps (Section 4.3).
      - Distill rationales and chain-of-thought to transfer reasoning capabilities with fewer samples (Section 4.3).

- Section 5: Evaluation (settings, datasets, metrics)
  - Table 2 maps settings → constraints → datasets → metrics. Examples:
    - Efficient inference: tasks like QA and NLU; metrics include inference time and throughput (Table 2; Section 5.2).
    - On-device/mobile: memory footprint/peak usage and compression ratio; datasets such as `TinyBERT` and `OpenOrca` subsets for evaluation (Table 2; Section 5.1).
    - Privacy-preserving: `PrivacyGLUE`, `MIMIC`; report privacy budget and noise levels to quantify privacy–utility trade-offs (Table 2).
    - Energy-efficient AI: report energy efficiency ratios, thermal efficiency, and idle power (Table 2; Section 5.2).

- Section 6: Applications (organized by constraints)
  - Real-time interaction: `GPT-4o`, `LLaMA-Omni`, EMOVA, Project Astra demonstrate low-latency, multi-modal, speech-aware experiences (Section 6.1).
  - Content generation/processing: multi-agent pipelines (e.g., Scene Analyzer GPT, Builder GPT) orchestrate analysis and code generation for MR/VR authoring (Section 6.2).
  - Edge inference and privacy: `MobileLLM`, on-device `Apple Intelligence` (≈3B), domain-adapted medical models, and MoE strategies for sparse compute on edge devices (Section 6.3).

- Sections 7–9: Open problems, conclusions, limitations
  - Hallucination, bias, energy, privacy are highlighted as persistent challenges with SLM-specific nuances (Sections 7, 9).

Definitions for uncommon terms used above:
- `KV cache`: cached keys/values from self-attention for previously generated tokens to avoid recomputing attention over the full history at every step.
- `n:m pruning`: a hardware-aligned pattern where exactly n weights out of each group of m are zeroed, enabling efficient sparse kernels.
- `VQ-VAE`: a vector-quantized autoencoder that maps continuous inputs (e.g., images) into discrete tokens.
- `BFLOAT16`/`FP8`: lower-precision floating-point formats that reduce compute and memory; BFLOAT16 has a wider exponent range than FP16; FP8 is an even lower-precision format used on newer GPUs.
- `ZeRO`/`FSDP`: sharded data-parallel techniques that partition optimizer states, gradients, and parameters across devices to reduce memory.
- `LoRA`: low-rank adapters trained while freezing the main weights.
- `QAT`: quantization-aware training that simulates quantization during training to improve post-quantization accuracy.
- `LSH`: locality-sensitive hashing, a fast method to find approximate nearest neighbors.

## 4. Key Insights and Innovations
- A constraint-centric taxonomy that is actionable (fundamental innovation)
  - What’s new: A two-axis map linking “where the method acts” (architecture/training/compression) to “which constraint it primarily optimizes” (Table 1; Table 2).
  - Why it matters: Practitioners can choose techniques tailored to their bottlenecks—e.g., if peak memory is the limiter, prefer quantization/pruning and efficient attention; if training compute is the limiter, employ mixed-precision, sharding, PEFT, and data augmentation.

- Clarification of trade-offs rather than one-size-fits-all (conceptual insight)
  - The survey emphasizes that gains in one dimension can cost another:
    > “It is important to note that progress on any one of these goals does not necessarily imply progress on the others.” (Section 1)
  - Example: QAT lowers memory and enables low-precision deployment but can be slower than full-precision training (Section 1; 4.2).

- Concrete mechanisms that repeatedly show up across SLMs (synthesis insight)
  - Hardware-aligned structure pays off: `n:m` pruning and grouped-query attention map to efficient kernels (Sections 2.1, 4.1).
  - Managing outliers is crucial for activation quantization: `SmoothQuant` and `SpinQuant` are general-purpose tools (Section 4.2).
  - KV-cache memory is a central bottleneck for long contexts; quantizing it is necessary for practical long-sequence SLMs (Section 4.2).

- Multimodal SLMs without heavyweight encoders (emerging capability)
  - Monolithic multimodal designs replace full vision encoders with tokenizers/MLPs, maintaining end-to-end smallness (Section 2.4).

## 5. Experimental Analysis
Because this is a survey, it does not present new controlled experiments; rather, it curates evaluation settings (Table 2) and reports representative empirical outcomes from prior work. The analysis here reflects what the survey highlights and how convincingly it connects methods to measured benefits.

- Evaluation methodology summarized by the survey:
  - Settings and constraints:
    - Efficient inference: evaluate latency via inference time and throughput on standard QA/NLU datasets such as SuperGLUE, SQuAD, TriviaQA, CoQA, and NQ (Table 2; Section 5.1–5.2).
    - On-device/mobile: measure peak memory usage, memory footprint, compression ratio; evaluate with small-scale NLU and instruction-following subsets (Table 2).
    - Privacy-preserving: use privacy-aware datasets (`PrivacyGLUE`, `MIMIC`), report privacy budgets (ε) and noise levels to show privacy–utility trade-offs (Table 2).
    - Energy-efficient AI: report energy efficiency ratio, thermal efficiency, idle power to connect model choices to power draw (Table 2).
  - This mapping is practical: it tells readers what to measure for each deployment goal and which datasets to use.

- Representative quantitative outcomes highlighted:
  - Lightweight architectures:
    > “MobileBERT... achieving a 4.3x size reduction and a 5.5x speedup compared to the base version of BERT.” (Section 2.1)
  - Efficient attention:
    > “Reformer... improves the complexity of the self-attention from O(N^2) to O(N log N).” (Section 2.2)
    - Linear attention mechanisms reduce to O(N), sometimes with RNN-like interpretations enabling faster inference (Section 2.2).
  - Compression:
    - `Wanda` avoids weight updates during pruning by using activation-aware criteria (Section 4.1).
    - `SmoothQuant` and `SpinQuant` explicitly address activation outliers (Section 4.2).
    - KV-cache quantization targets long-context feasibility (Section 4.2).

- Do these support the claims?
  - The survey does not aggregate new benchmark tables; instead, it grounds claims by pointing to canonical works and their reported gains (e.g., MobileBERT’s speedup, Reformer’s asymptotic cost reduction).
  - Its most convincing evidence is the alignment between mechanisms and constraints (Tables 1–2), plus concrete examples where specific methods yield known improvements (e.g., FlashAttention in TinyLLaMA, Section 2.1).

- Ablations, failure cases, robustness:
  - Not applicable as no new experiments are run. However, the survey partially addresses robustness/deployment issues by emphasizing:
    - Trade-offs (QAT speed vs. memory; unstructured sparsity requiring specialized kernels) (Section 1; 4.1–4.2).
    - KD caveats (tokenizer mismatch, lack of teacher pretraining data) and solutions (universal logit distillation) (Section 4.3).

- Conditional/mixed results and trade-offs:
  - Many methods depend on hardware/software stack maturity (e.g., `n:m` sparsity benefits are unlocked by frameworks like TensorRT) (Section 4.1; Appendix A).
  - PEFT and data augmentation reduce finetuning cost but depend on high-quality synthetic data and calibration (Section 3.2).

## 6. Limitations and Trade-offs
- Scope and assumptions:
  - “Smallness” is context-dependent; there is no universal parameter threshold (Section 1).
  - The taxonomy marks “primary” constraints addressed by each technique (Table 1), but techniques can impact multiple dimensions with complex interactions (Section 1).

- Technical trade-offs and constraints:
  - Mixed precision and `QAT`:
    - Gains in memory/compute may come with slower wall-clock training due to extra bookkeeping or simulated quantization (Section 1; 4.2).
  - Sparsity:
    - Unstructured pruning often needs specialized sparse kernels/hardware to realize speedups; otherwise, it mainly reduces storage without runtime gains (Section 4.1; Appendix A).
  - KD:
    - Requires careful handling of tokenizer mismatch and teacher–student data alignment; access to a strong teacher and/or its data can be a barrier (Section 4.3).
  - NAS:
    - Search remains costly for >1B parameter regimes; methods therefore rely on reduced spaces or heuristics (Section 2.3).

- Scenarios not addressed or underexplored:
  - Unified benchmarking: the survey outlines datasets/metrics but does not propose a single standardized SLM benchmark suite combining accuracy, latency, energy, and privacy simultaneously (Section 5).
  - Security aspects (prompt injection, model extraction) are mentioned but not comprehensively treated as design constraints (Section 7.4).

- Compute/data/hardware dependencies:
  - Many wins are contingent on modern GPU features (e.g., `FlashAttention`, `FP8`, TensorRT sparsity support) and may not transfer 1:1 to CPU-only or microcontroller-class devices (Sections 2.1, 3.1, 4.1).

## 7. Implications and Future Directions
- How this work changes the landscape:
  - It reframes SLM development as a constraint-optimization problem with a toolkit spanning architecture, training, and compression, enabling practitioners to assemble targeted “recipes” for specific deployment settings (Tables 1–2).
  - It spotlights mechanisms (attention approximations, KV-cache quantization, activation outlier handling, dynamic adapters) that are maturing into standard SLM ingredients.

- Follow-up research enabled/suggested:
  - Integrated, hardware-aware pipelines:
    - Combine structured pruning (`n:m`) with quantization and KV-cache compression optimized for a specific device class (GPU, NPU, CPU-only edge) and workload (Section 4.1–4.2).
  - Standardized SLM benchmarks:
    - A combined accuracy–latency–memory–energy–privacy suite for realistic edge scenarios would anchor progress (Table 2).
  - Better long-context SLMs:
    - Further K/V cache innovations, approximate attention with verifiable accuracy bounds, and training curricula that maintain long-range reasoning (Section 2.2; 4.2).
  - Robust distillation:
    - Tokenizer-agnostic distillation losses and rationale/chain-of-thought distillation scaled to multi-domain students (Section 4.3).
  - Multimodal monolithic models:
    - Lightweight tokenizers and modality-specific experts that match larger encoders on core perception tasks while staying sub-billion (Section 2.4).
  - Safety, energy, and privacy by design:
    - Techniques that jointly reduce hallucination, bias, and energy, with measurable privacy guarantees (Sections 7.1–7.4; Table 2).

- Practical applications and use cases:
  - On-device assistants:
    - Offline summarization, notification triage, speech interfaces with low latency and privacy, as in `Apple Intelligence` and `TalkBack with GeminiNano` (Section 6.3).
  - Domain-specific SLMs:
    - Clinical documentation assistance (`MIMIC`, domain-adapted LMs), legal summarization, and low-resource language services where privacy and cost dominate (Sections 5.1, 6.3).
  - Edge multimodal analytics:
    - Real-time perception + language reasoning for AR/VR and robotics (Section 6.1–6.2).

> Summary quote of the paper’s aim:
> “Our survey aims to serve as a valuable resource for researchers and practitioners interested in developing and deploying small yet efficient language models.” (Abstract)

> And its organizational backbone:
> “An overview of these axes can be found in Table 1 (techniques) and Table 2 (constraints).” (Section 1)

Together, the taxonomy (Tables 1–2), the mechanism-first explanations (Sections 2–4), and the deployment-oriented applications and open challenges (Sections 6–7) provide a practical blueprint for building SLMs that meet real-world constraints without sacrificing capability.
