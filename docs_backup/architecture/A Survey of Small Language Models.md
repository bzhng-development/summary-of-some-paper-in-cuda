# A Survey of Small Language Models

**ArXiv:** [2410.20011](https://arxiv.org/abs/2410.20011)
**Authors:** Chien Van Nguyen, Xuan Shen, Ryan Aponte, Yu Xia, Samyadeep Basu, Zhengmian Hu, Jian Chen, Mihir Parmar, Sasidhar Kunapuli, Joe Barrow, Junda Wu, Ashish Singh, Yu Wang, Jiuxiang Gu, Franck Dernoncourt, Nesreen K. Ahmed, Nedim Lipka, Ruiyi Zhang, Xiang Chen, Tong Yu, Sungchul Kim, Hanieh Deilamsalehy, Namyong Park, Mike Rimer, Zhehao Zhang, Huanrui Yang, Ryan A. Rossi, Thien Huu Nguyen
**Institutions:** 

## 🎯 Pitch

This paper introduces a novel two-axis taxonomy to navigate the burgeoning field of Small Language Models (SLMs), linking specific model techniques to real-world hardware and system constraints like memory, latency, and energy efficiency. By systematically mapping techniques such as compression and training optimizations to these constraints, the paper empowers practitioners to tailor SLMs for edge and on-device environments, thereby enhancing privacy, performance, and accessibility in resource-constrained settings.

---

## 1. Executive Summary (2-3 sentences)
This survey organizes the rapidly growing space of Small Language Models (SLMs) into a practical, two-axis taxonomy that maps concrete techniques (architectures, training, compression) to the hardware and system constraints they optimize (e.g., memory, latency, energy) (Table 1 and Table 2). It explains how to build, train, compress, and evaluate SLMs, and highlights open challenges (hallucination, bias, privacy, energy) that matter for deploying models on-device, at the edge, and in resource-constrained settings.

## 2. Context and Motivation
- Problem addressed
  - Large Language Models (LLMs) are expensive to train and serve; they typically require specialized, centralized hardware and large memory/compute resources (Section 1).
  - There is increasing demand for models that retain useful capability under strict constraints (limited memory, latency budgets, energy/battery, privacy, bandwidth), so they can run on mobile, edge, and embedded devices (Section 1).
  - The definition of “small” is fluid and context-dependent, which complicates benchmarking and design choices over time and across hardware (Section 1).

- Why this is important
  - Real-world impact: SLMs enable on-device functionality (faster time-to-first-token, offline availability), lower costs, and better privacy by avoiding cloud data transfer (Section 6.3).
  - Theoretical/practical significance: Different efficiency techniques often trade off against each other; clear guidance is needed to match methods to constraints (Section 1).
    > “It is important to note that progress on any one of these goals does not necessarily imply progress on the others… memory-efficient training methods like quantization-aware training are often slower than their full-precision counterparts.” (Section 1)

- Prior approaches and their limitations
  - Existing surveys largely focus on LLMs, alignment, and learning paradigms (e.g., Rogers et al., 2020; Shen et al., 2023) rather than SLMs and their constraint-driven design (Section 1).
  - Fragmented landscape: architectural tricks, training practices, and compression methods are often discussed separately, making it hard to compose them coherently for a given device or constraint.

- How this paper positions itself
  - Provides the first SLM-focused survey, introducing a two-axis taxonomy:
    - Techniques across the model lifecycle (architectures → training → post-training compression) (Table 1).
    - The constraints each technique targets (inference compute, memory, latency, storage, training compute/time, energy, and privacy) (Table 1 and Table 2).
    > “An overview of these axes can be found in Table 1 (techniques) and Table 2 (constraints).” (Section 1)
  - Synthesizes concrete mechanisms (e.g., linear attention, mixed precision, structured pruning, K/V-cache quantization) and maps them to outcomes that practitioners care about (Sections 2–4, Tables 1–2).

## 3. Technical Approach
This is a survey, not a new algorithm. Its “methodology” is a structured framework that guides SLM builders from constraints to techniques to concrete mechanisms:

- Step 1 — Identify the operational constraint(s) (Table 2)
  - Examples: latency/throughput for real-time apps; memory/storage for on-device; privacy constraints; energy efficiency for battery-powered devices (Section 5.2; Table 2).
  - Metrics are aligned to constraints:
    - Latency: inference time and throughput (Section 5.2).
      > “Inference time measures how quickly a model can process input… Throughput… the number of tokens or samples a model can process…” (Section 5.2)
    - Memory: peak memory usage, memory footprint, compression ratio (Section 5.2).
    - Privacy: privacy budget and noise level (Section 5.2).
    - Energy: energy efficiency ratio, thermal efficiency, idle power (Section 5.2).

- Step 2 — Choose lifecycle stage(s) to intervene (Table 1)
  - Model architectures (pre-processing): lightweight designs, efficient attention, or neural architecture search (Section 2).
  - Training: pretraining efficiency (mixed precision, distributed training) and fine-tuning (PEFT, data augmentation) (Section 3).
  - Compression (post-processing): pruning, quantization, knowledge distillation (Section 4).
  - Table 1 associates each bucket with the constraint(s) it most directly improves.

- Step 3 — Apply concrete mechanisms (Sections 2–4)
  - Architectures (Section 2)
    - Lightweight encoder-only models: `MobileBERT` uses an inverted bottleneck (widen then narrow) to balance attention and feed-forward networks, achieving 4.3× smaller size and 5.5× speedup relative to BERT-base (Section 2.1).
    - Lightweight decoder-only models: `BabyLLaMA` and `BabyLLaMA-2` distill from multiple teachers into 58M and 345M parameter models; `TinyLLaMA` (1.1B) uses memory-aware kernels like `FlashAttention`; `MobileLLM` uses embedding sharing, grouped-query attention (GQA: fewer key/value heads per multiple query heads), and block-wise weight sharing to reduce latency and memory (Section 2.1).
    - Efficient attention approximations (Section 2.2):
      - `Reformer`: uses locality-sensitive hashing (LSH) to retrieve likely-relevant keys, reducing attention complexity from O(N^2) to O(N log N). Mechanism: hash queries/keys into buckets and compute attention within buckets.
        > “Reformer… improves the complexity of the self-attention from O(N^2) to O(N log N)… using locality-sensitivity hashing.” (Section 2.2)
      - Linear attention: approximate softmax attention via kernel feature maps so attention can be computed as two linear passes (e.g., `Linformer`, `Performer`-like ideas), reducing to O(N) time/space under assumptions about low-rank or kernelizable attention (Section 2.2).
      - State-space models (`Mamba`), and hybrids like `RWKV`: replace attention with selective state transitions that scale linearly in sequence length while retaining long-range modeling via learned dynamics (Section 2.2).
      - Long context encoders: `Longformer` uses sliding-window attention plus task-specific global tokens for linear scaling; `Nyströmformer` uses the Nyström method to approximate softmax attention via landmark points (Section 2.2).
    - Neural Architecture Search (NAS): automated search is hard at LLM scale due to cost; targeted depth/width sweeps in the millions-of-parameters regime and better search initialization shrink the search space (Section 2.3).

  - Training (Section 3)
    - Mixed precision pretraining:
      - `FP16`/`BFLOAT16`: do forward/backward in low precision while keeping an FP32 master copy for updates; `BFLOAT16` keeps a larger exponent for better numerical range (Section 3.1).
      - `FP8` on NVIDIA Hopper: hardware support lets you move more computation into 8-bit floating point for speed/energy gains (Section 3.1).
    - Optimizers and stability: memory-efficient optimizers like `Adafactor` and `Sophia`, gradient clipping, and careful initialization to reduce instabilities and memory use (Section 3.1).
    - Distributed training:
      - `ZeRO` stages 1–3: progressively shard optimizer states, gradients, and parameters across workers; `FSDP` is the PyTorch counterpart (Section 3.1).
    - Parameter-efficient fine-tuning (PEFT) (Section 3.2.1):
      - `LoRA`: injects trainable low-rank matrices into frozen weight matrices; only small adapters are updated.
      - Prompt-tuning: learn a small set of input embeddings (“soft prompts”) instead of updating the model.
      - `LLaMA-Adapter`: adds small prompt modules to attention blocks.
      - Dynamic adapters (mixture-of-experts adapters): combine multiple small adapters and route per input/task to avoid forgetting and enable multitask usage.
    - Data augmentation for fine-tuning (Section 3.2.2):
      - Synthetic data via paraphrasing, instruction evolution, reflection-based refinement, retrieval-augmented generation, and hard-example mining with LLM feedback; particularly useful for low-resource, medical, and privacy-sensitive settings.

  - Compression (Section 4; Appendix A)
    - Pruning (Section 4.1):
      - Unstructured pruning (delete individual weights): e.g., `SparseGPT` treats pruning as a layer-wise sparse regression to minimize error; `Wanda` scores weights using activations and avoids retraining; `n:m` structured sparsity within small blocks (e.g., 2:4) maps better to GPU kernels (Appendix A).
      - Structured pruning (remove neurons/heads/layers): exploit redundancy in feed-forward blocks and layers; switch activations to `ReLU` to increase activation sparsity (Section 4.1).
      - Input-dependent pruning: “contextual sparsity” prunes per-input, or compresses K/V cache adaptively (`FastGen`) to save inference memory (Section 4.1).
    - Quantization (Section 4.2):
      - Weight-only: `GPTQ` quantizes weights per-layer via an approximate inverse-Hessian objective to minimize output error.
      - Weight+activation: `AWQ`, `ZeroQuant` consider activations to better preserve important weight scales for efficient INT GEMMs.
      - K/V cache quantization: directly quantize the attention cache (keys/values) to support long context without linear growth in memory.
      - Handling activation outliers: `SmoothQuant` scales channels to shift difficulty from activations to weights; `SpinQuant` rotates vectors to distribute outliers across dimensions before quantization.
      - Quantization-aware training (QAT): fine-tune with fake-quantized weights/activations (sometimes distilling from an FP16 teacher) to recover quality lost in post-training quantization; practical variants target edge devices and FPGAs (Section 4.2).
    - Knowledge distillation (KD) (Section 4.3):
      - Classical KD: train a small “student” to match the probability distribution (logits) of a large “teacher”.
      - Enhancements: loss shaping (e.g., f-divergence), layer-wise task-aware distillation, multi-teacher fusion (merge teacher distributions), cross-tokenizer distillation via optimal-transport-inspired losses, and distilling intermediate reasoning (“rationales,” chain-of-thought) for better sample efficiency and reasoning ability.
      - Joint prune–distill cycles can yield strong compact models in practice.

- Small multimodal SLMs (Section 2.4)
  - Use smaller LMs (e.g., `Gemma`, `phi-3-mini`) and lighter visual stacks.
  - Reduce vision encoder size by truncating deeper blocks or exploiting intermediate features; “monolithic” models convert images to discrete tokens via `VQ-VAE` or learned lightweight tokenizers and feed directly to the LM (e.g., `Chameleon`, `Mono-InternVL`).

## 4. Key Insights and Innovations
- A unifying, constraint-first taxonomy (Tables 1–2)
  - What’s new: The survey explicitly ties each technique family (architecture, training, compression) to the constraints it primarily improves (latency, memory, storage, training compute, energy, privacy) (Table 1, Table 2).
  - Why it matters: Practitioners can choose interventions that directly address their deployment bottlenecks rather than applying generic “efficiency” methods.

- Clear mechanism-level explanations of “how to get O(N)” sequence processing (Section 2.2)
  - What’s highlighted: The paper traces a path from sparse/hashed attention (`Reformer`) to kernelized linear attention (`Linformer`-style), to state-space models (`Mamba`), and hybrid RNN–Transformer designs (`RWKV`), all of which aim for linear time/space.
  - Why it matters: This helps readers understand when to favor attention approximations versus state-space dynamics for long sequences and small memory budgets.

- Practical, composable post-training playbook (Section 4; Appendix A)
  - What’s emphasized: Complementary, not competing, techniques—e.g., combine weight+activation quantization with K/V-cache quantization; prune structurally to map to hardware; add QAT or KD to recover accuracy.
  - Why it matters: SLMs often need multiple small gains that add up; the survey shows what stacks well together, and where hardware support (e.g., `n:m` sparsity, INT GEMMs) matters.

- Evaluation is constraint-driven, not benchmark-driven (Section 5; Table 2)
  - What’s new: Metrics are organized by the real-world constraint they quantify (latency, memory, privacy, energy) rather than by task family.
  - Why it matters: You evaluate what you intend to optimize. This helps teams select appropriate datasets/metrics (e.g., privacy budgets for medical assistants; energy efficiency for mobile).

- Spotlight on small multimodal stacks (Section 2.4; Section 6)
  - What’s highlighted: Strategies to shrink the vision side (truncate encoders, use lightweight encoders, or tokenizers) enable end-to-end multimodal SLMs that fit device limits.
  - Why it matters: Many new on-device assistants are multimodal (voice, image), so SLMs must include efficient non-text components.

Overall, these are organizing and synthesis innovations rather than a single new algorithm—valuable for practitioners navigating a complex design space.

## 5. Experimental Analysis
This survey does not run new experiments; instead, it curates methods, metrics, and representative results from the literature. Where the paper reports numbers, they are used to illustrate mechanisms and trade-offs.

- Evaluation methodology summarized (Section 5; Table 2)
  - Datasets by constraint:
    - Efficient inference: SuperGLUE, SQuAD, TriviaQA, CoQA, Natural Questions—tasks where response speed matters.
    - Privacy-preserving: PrivacyGLUE, MIMIC, n2c2, LEAF (federated)—tasks/data requiring anonymity or data locality.
    - On-device/TinyML: TinyBERT and OpenOrca subsets—designed/curated for training/evaluating small models (Section 5.1).
  - Metrics by constraint (Section 5.2):
    > Latency: “Inference time… and Throughput…”  
    > Memory: “Peak memory usage… Memory footprint, Compression ratio…”  
    > Privacy: “Privacy budget… Noise level…”  
    > Energy: “Energy Efficiency Ratio… Thermal Efficiency, Idle Power Consumption…”

- Representative quantitative examples embedded in the survey
  - `MobileBERT`: 4.3× size reduction and 5.5× speedup vs. BERT-base through inverted-bottleneck design (Section 2.1).
  - `BabyLLaMA`: 58M parameters; `BabyLLaMA-2`: 345M parameters—distillation can outperform teachers in low-data settings (Section 2.1).
  - `TinyLLaMA`: 1.1B parameters with memory-efficient attention kernels—competitive downstream performance at small scale (Section 2.1).

- Mechanism-level analyses (Sections 2–4)
  - Attention complexity reductions:
    > “Reformer… O(N log N)… [via] locality-sensitivity hashing.” (Section 2.2)  
    > Linear attention approximates softmax via kernels or low-rank structure to reach O(N).  
  - Quantization challenges and remedies:
    - Activation outliers (rare, large values) can break low-bit quantization; `SmoothQuant` shifts scale to weights; `SpinQuant` uses learned rotations (Section 4.2).
  - Pruning practicality:
    - Unstructured sparsity requires specialized kernels/hardware to realize speedups, while structured sparsity (e.g., `n:m`) aligns better with acceleration libraries (Section 4.1; Appendix A).

- Do the curated results support the claims?
  - The survey carefully ties mechanisms to reported outcomes (speedups, memory reductions, accuracy retention) and flags trade-offs (e.g., QAT can be slower even if memory-friendly) (Section 1; Sections 4.1–4.2).
  - It also points to implementation evidence (e.g., mobile/FPGA deployments for quantized models; NVIDIA TensorRT exploiting `n:m` patterns) (Section 4.2; Appendix A).

- Robustness, failure modes, and trade-offs
  - Trade-off example:
    > “Quantization-aware training… is often slower than… full-precision… [but] allow[s] training or finetuning using less memory.” (Section 1)
  - Attention approximations often assume low-rank or locality structure; their benefits depend on sequence characteristics and task demands (Section 2.2).
  - Distillation effectiveness can depend on tokenizer compatibility and access to (or similarity with) the teacher’s pretraining distribution (Section 4.3).

In short, the survey’s “experimental” contribution is a constraint-centric evaluation map (datasets, metrics) and well-chosen quantitative examples that illustrate how each mechanism achieves its claimed efficiency.

## 6. Limitations and Trade-offs
- Fuzzy definition of “small” (Section 1)
  - The notion of “small” depends on context and time (e.g., GPT-2 1.5B once “large” now exceeds some SLMs), complicating universal benchmarks and claims.

- Hardware-dependence of speedups (Section 4.1; Appendix A)
  - Unstructured sparsity rarely accelerates without specialized runtimes; structured patterns like `n:m` and block sparsity are easier to accelerate (but may prune less flexibly).
  - Quantization acceleration depends on INT kernels and hardware support (e.g., mobile NPUs, Tensor Cores).

- Training-time vs inference-time trade-offs (Sections 3.1, 4.2)
  - QAT and KD can recover quality but add training cost and complexity.
  - Mixed-precision and distributed training reduce memory/compute but introduce new stability and engineering challenges (e.g., loss scaling, sharding strategies).

- Distillation assumptions (Section 4.3)
  - Many distillation techniques assume compatible tokenizers and/or access to teacher-like data. Cross-tokenizer KD and data-free approaches exist but are less mature.

- NAS at LLM scale is still costly (Section 2.3)
  - Search is mostly feasible in the sub-billion parameter regime or with tightly scoped depth/width searches; general LLM NAS remains expensive.

- Open risks (Section 7)
  - Hallucination and bias persist; their relationship to model size is nontrivial (some hallucinations decrease with size, some biases increase) (Sections 7.1–7.2).
  - Privacy risks span training data, system prompts (prompt leakage/abuse), and user queries at inference (Section 7.4).
  - Energy profiling is nascent and sensitive to hardware, workload mix, and response length (Section 7.3).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a practical blueprint: start from constraints and pick mechanisms that directly target them (Tables 1–2). This reframes “efficiency” as an application-driven design process rather than a grab-bag of tricks.

- Follow-up research enabled or suggested
  - Standardized, constraint-aligned benchmarks
    - More public suites measuring latency/throughput, peak memory, energy, and privacy budgets across common edge hardware.
  - Composable toolchains
    - End-to-end pipelines that combine pruning→quantization→KD with hardware-aware compilation and K/V-cache optimizations.
  - Robust attention alternatives
    - Continued work on SSMs/linear attention with clearer guidance on when they beat classic attention (sequence characteristics, training stability).
  - Distillation without data/compatibility hurdles
    - Broader cross-tokenizer KD, universal logit objectives, and rationale/chain-of-thought distillation tailored for tiny models.
  - Energy-first modeling
    - Methods that explicitly trade response length, decoding strategy, and adapter routing to minimize energy (Section 7.3).
  - Privacy-by-design SLMs
    - On-device training/fine-tuning with formal privacy guarantees; mitigation of prompt leakage/abuse in agent-like systems (Section 7.4).

- Practical applications and use cases (Section 6; Table 3)
  - Real-time assistants: speech and multimodal interaction (`GPT-4o`, `LLaMA-Omni`, EMOVA, Project Astra) where low latency and on-device compute are vital (Section 6.1).
  - Content generation/editing in constrained environments: VR programming assistants (`LLMR`, `DreamCodeVR`) (Section 6.2).
  - Edge inference and privacy-sensitive domains: on-device chat, summarization, biomedical assistants (`MobileLLM`, Apple Intelligence, `HuatuoGPT`, `BioMistral`), accessibility tools like `TalkBack` with `GeminiNano` (Section 6.3).
  - Energy- and cost-aware inference: Mixture-of-Experts routing (e.g., `EdgeMoE`) to activate only subsets of parameters per input (Section 6.3).

Selected anchoring quotes for key claims:
- > “We propose a novel taxonomy for categorizing the methods used to optimize SLMs… We summarize the benchmark datasets… [and] highlight key open challenges.” (Abstract)
- > “An overview of these axes can be found in Table 1 (techniques) and Table 2 (constraints).” (Section 1)
- > “Reformer… improves the complexity… from O(N^2) to O(N log N)… [via] locality-sensitivity hashing.” (Section 2.2)
- > “Appendix A provides further discussion of pruning techniques.” (Section 4.1)
- > “The key metrics… are… [Latency] Inference time, Throughput… [Memory] Peak memory usage… [Privacy] Privacy budget, Noise level… [Energy] Energy Efficiency Ratio…” (Section 5.2)
- > “Hallucination can be classified into… factuality and faithfulness.” (Section 7.1)

In sum, this survey equips readers with a constraint-first mental model, concrete mechanisms, and an evaluation map to design and deploy capable SLMs under real-world limits.
