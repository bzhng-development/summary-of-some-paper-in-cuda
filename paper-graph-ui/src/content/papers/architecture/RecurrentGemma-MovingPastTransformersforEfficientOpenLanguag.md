# RecurrentGemma: Moving Past Transformers for Efficient Open Language Models

**ArXiv:** [2404.07839](https://arxiv.org/abs/2404.07839)

## 🎯 Pitch

RecurrentGemma unveils a new family of open language models leveraging Google's Griffin architecture, which breaks from the Transformer paradigm by combining linear recurrences with local attention to achieve fixed-size memory during inference. This innovation delivers Transformer-comparable accuracy but with dramatically improved inference speed and efficiency, enabling long-sequence processing on smaller hardware—a breakthrough for scalable, resource-constrained AI applications that require handling lengthy or streaming text.

---

## 1. Executive Summary
RecurrentGemma introduces two open language models (`RecurrentGemma-2B` and `RecurrentGemma-9B`) built on Google’s `Griffin` architecture, which mixes linear recurrences with local attention to keep a fixed-size state during inference. This design delivers Transformer‑level accuracy with substantially faster and more memory‑efficient inference on long sequences, enabling smaller hardware to handle long text generation without the escalating memory costs of a Transformer KV cache (Figures 1a–1b, Table 2).

## 2. Context and Motivation
- Problem addressed
  - Inference for Transformer language models is memory-bound because each generation step must retrieve and update the per-layer `KV cache` (stored Keys and Values for attention), which grows linearly with sequence length. This growth slows generation and caps maximum length by device memory.
  - Using local attention in Transformers reduces the cache size but typically hurts task performance.

- Why this matters
  - Real-world applications (long-form writing, code generation, log analysis, interactive assistants) need efficient long-sequence inference. Reducing memory pressure and keeping throughput high as sequences grow lowers serving costs and enables on-device or modest-infrastructure deployment.

- Prior approaches and shortcomings
  - Standard Transformers with global attention deliver high accuracy but scale memory with sequence length.
  - Local-attention Transformers reduce memory but often at a performance cost (Introduction).
  - Multi-Query Attention (shared Keys/Values across heads) helps reduce KV memory compared to Multi-Head Attention, but the cache still grows with sequence length.

- Positioning of this work
  - RecurrentGemma adopts `Griffin`—a mixture of linear recurrences and local attention—to compress the past into a fixed-size recurrent state while retaining high accuracy on language tasks.
  - The paper releases pre-trained and instruction-tuned checkpoints at 2B and 9B parameters, along with JAX/Flax and reference PyTorch implementations and a TPU‑optimized recurrence kernel (Pallas), and directly compares throughput and accuracy to similarly sized Gemma Transformers (Introduction; Figures 1a–1b; Table 2).

## 3. Technical Approach
Step-by-step overview
- Core architecture: Griffin
  - Griffin models each token using two pathways:
    1) A `linear recurrence` pathway that maintains a compact hidden state summarizing the entire processed history. A linear recurrence updates the state with a matrix‑based rule that depends linearly on the previous state and current input; it functions like a lightweight, gated “memory” that does not grow with sequence length.
    2) A `local attention` pathway that attends only within a fixed window (here, 2,048 tokens; Table 1).
  - By combining these, the model keeps a constant-size recurrent state plus a small local window, avoiding a full-length KV cache (Introduction; Table 1).

- Paper-specific modifications to Griffin (Model architecture)
  - Multiply input embeddings by `sqrt(model width)` before entering the network. The input and output embeddings are tied, but the scaling is not applied at the output.
  - Do not apply weight decay to parameters of the recurrent (`RG-LRU`) layers during training. `RG-LRU` can be read as the specialized linear-recurrence block used in Griffin.
  - Clip the gradient through the square-root operation in recurrent layers to a maximum of `1000` to stabilize training.

- Model sizes and hyperparameters (Table 1)
  - `RecurrentGemma-2B`
    - Total params: 2.68B (2.03B non‑embedding; 0.65B embedding)
    - Width: 2560; Depth: 26; MLP expansion: 3
    - Attention heads: 10; Local attention window: 2048
    - Vocabulary: 256k
  - `RecurrentGemma-9B`
    - Total params: 8.58B (7.53B non‑embedding; 1.05B embedding)
    - Width: 4096; Depth: 38; MLP expansion: 3
    - Attention heads: 16; Local attention window: 2048
    - Vocabulary: 256k
  - Note: The 256k vocabulary makes embeddings a large fraction of total parameters.

- Training pipeline (Training details: Pre-training)
  - Sequence length: 8,192 tokens.
  - Data: same high‑level mixture as Gemma (web documents, math, code) with filtering for safety, sensitive data, and removal of eval sets.
  - Tokenization: SentencePiece subset with a 256k vocab.
  - Regimen: pre-train both `2B` and `9B` on 2T tokens using a two‑stage schedule (broad mixture, then smaller high‑quality set). For comparison, Gemma-2B and Gemma-7B used 3T and 6T tokens respectively (Table 2 notes).

- Instruction tuning and RLHF (Instruction tuning and RLHF)
  - Format control tokens to structure dialogue during supervised fine-tuning (SFT) and RLHF (Table 3), e.g.:
    - `user`, `model`, `<start_of_turn>`, `<end_of_turn>`.
  - Example (Table 4) shows how these tokens wrap turns to make the dialogue format explicit.
  - They use a “novel RLHF algorithm” (as in the Gemma report) to increase the reward of generated responses, but detailed algorithmic specifics are not elaborated here.

- Inference and implementation
  - The fixed-size recurrent state means memory does not grow beyond the local window (2K tokens). This is crucial for long-sequence generation.
  - JAX/Flax implementation includes a `Pallas` kernel—custom low-level code specialized for TPUs to accelerate the linear recurrence (Inference Speed Benchmarks). A reference PyTorch implementation is provided; throughput on GPUs is expected to be lower.

- How this yields efficiency
  - Transformers: every new token requires consulting the full KV cache whose size grows with sequence length; generation throughput typically falls as sequences get longer.
  - RecurrentGemma: after the local window is established, only the fixed recurrent state (plus local attention window) is used; throughput stays roughly constant as sequences grow (Figures 1a–1b).

## 4. Key Insights and Innovations
- Fixed-size state with Transformer-like accuracy (fundamental innovation)
  - Mixing linear recurrence with local attention keeps memory constant w.r.t. sequence length beyond the local window while maintaining accuracy competitive with Transformers of similar size (Introduction; Table 2).
  - Significance: enables arbitrarily long generations without running out of memory, and stabilizes throughput at long lengths (Figures 1a–1b).

- Token‑efficient training at small scale (practical insight)
  - With only 2T pre-training tokens, `RecurrentGemma-2B` is comparable to `Gemma-2B` trained on 3T, and `RecurrentGemma-9B` is comparable to `Gemma-7B` trained on 6T (Table 2).
  - This suggests the Griffin design can reach strong accuracy with fewer tokens at these scales.

- Targeted stability choices for recurrence (incremental but important)
  - Three small but targeted modifications—input embedding scaling, no weight decay on RG‑LRU parameters, and gradient clipping through sqrt—address training stability and optimization for the recurrent pathway (Model architecture).

- Throughput that does not degrade with sequence length (fundamental operational advantage)
  - In sampling, RecurrentGemma’s throughput remains high and almost flat as sequence length increases, while Gemma’s declines as its KV cache grows. For `9B` vs `7B`, gains can reach “up to two orders of magnitude” in long generations (Figure 1b).
  - This is the main operational payoff of the fixed-state design.

## 5. Experimental Analysis
- Evaluation methodology
  - Automated benchmarks (Table 2): common academic tasks across knowledge, reasoning, commonsense, QA, and coding. Metrics include zero-shot and few-shot accuracy (e.g., MMLU 5-shot top‑1, HellaSwag 0-shot, HumanEval pass@1).
  - Human evaluation (Table 5): head-to-head win rates versus Mistral 7B v0.2 Instruct on ~1,000 instruction-following prompts and ~400 safety prompts. Wins/ties/losses are reported; ties count as 0.5 wins.
  - Inference speed (Figures 1a–1b): throughput on single devices (TPUv5e for 2B, TPUv4 for 9B) as a function of sample length from a 2K prompt, and as a function of prompt length when precomputing the initial state.
  - Safety benchmarks (Table 6): RealToxicity, BOLD, CrowS-Pairs, BBQ, Winogender, TruthfulQA, Winobias, Toxigen. Some are higher-better; others (RealToxicity, Toxigen) are lower-better.

- Main quantitative results
  - Accuracy vs Gemma (Table 2)
    - Averages:
      > RecurrentGemma-2B average: 44.6 vs Gemma-2B: 45.0  
      > RecurrentGemma-9B average: 56.1 vs Gemma-7B: 56.9
    - Selected tasks, `9B` vs `7B`:
      > MMLU (5-shot): 60.5 vs 64.3  
      > TriviaQA (5-shot): 70.5 vs 63.4  
      > BBH: 55.2 vs 55.1
    - Selected tasks, `2B` vs `2B`:
      > HellaSwag (0-shot): 71.0 vs 71.4  
      > PIQA (0-shot): 78.5 vs 77.3  
      > GSM8K (maj@1): 13.4 vs 17.7
    - Takeaway: At both sizes, RecurrentGemma closely tracks Gemma’s performance despite fewer training tokens (2T vs 3T/6T).

  - Human evaluation (Table 5)
    - Safety prompts:
      > RecurrentGemma-2B IT: 59.8% win rate (95% CI [57.1%, 62.6%])  
      > RecurrentGemma-9B IT: 59.9% win rate (95% CI [57.1%, 62.6%])
    - Instruction-following prompts:
      > RecurrentGemma-2B IT: 43.7% win rate vs Mistral 7B  
      > RecurrentGemma-9B IT: 59.3% win rate vs Mistral 7B (95% CI [57.4%, 61.2%])
    - Takeaway: The 9B instruction-tuned model substantially outperforms Mistral 7B on instruction following; even the 2B is competitive given its smaller size.

  - Inference speed (Figures 1a–1b)
    - Sampling throughput from a 2K prompt:
      > RecurrentGemma maintains high throughput as sample length increases, whereas Gemma slows down.  
      > For `2B`, RecurrentGemma sampling reaches about 6K tokens/sec; prompt processing is ~40K tokens/sec for both Gemma and RecurrentGemma (Figure 1a).  
      > For `9B`, throughput advantages over Gemma-7B can be “up to two orders of magnitude” at long lengths (Figure 1b).
    - Prompt processing:
      > Similar speeds across architectures because it is fully parallelizable: ~40K tok/s (2B), ~12K tok/s (9B).
    - Note: Measurements are on TPU with the Flax/Pallas kernel; GPU/PyTorch users should expect lower throughput (Inference Speed Benchmarks).

  - Safety benchmarks (Table 6)
    - RealToxicity (lower is better):
      > 2B: PT 9.8 → IT 7.6 (improves)  
      > 9B: PT 10.3 → IT 8.8 (improves)
    - Toxigen (lower is better):
      > 2B: PT 56.7 → IT 50.0 (improves)  
      > 9B: PT 58.8 → IT 64.5 (worsens)
    - TruthfulQA (higher is better):
      > 2B: PT 35.1 → IT 42.7 (improves)  
      > 9B: PT 38.6 → IT 47.7 (improves)
    - Takeaway: Instruction tuning generally improves safety and truthfulness metrics, with the notable exception of Toxigen for the 9B IT model.

- Do the experiments support the claims?
  - Efficiency: Yes. Figures 1a–1b directly show that RecurrentGemma’s sampling throughput does not degrade with longer sequences, while Gemma’s does.
  - Accuracy: Table 2 shows performance parity between RecurrentGemma and Gemma at similar parameter counts, despite fewer training tokens for RecurrentGemma.
  - Usability: Table 3–4 clarify formatting for instruction-tuned use; Table 5 demonstrates competitive human-judged utility.

- Notable gaps
  - No ablation isolating the impact of each architectural/training modification (embedding scaling, no weight decay on RG-LRU, gradient clipping).
  - No dedicated long-context task benchmarks (e.g., long-range retrieval) to empirically test whether the recurrence + local window substitutes fully for global attention across very long documents.

## 6. Limitations and Trade-offs
- Hardware dependence for peak speed
  - Throughput results are from TPU‑optimized Flax with a custom Pallas kernel; the paper cautions that GPU/PyTorch throughput will be lower (Inference Speed Benchmarks).

- Evaluation scope
  - The benchmark suite (Table 2) is broad but not specialized for long-context reasoning or retrieval; while throughput scales well with length, quality on tasks requiring dependencies beyond the local window is not directly evaluated here.

- Safety metrics are mixed
  - Instruction tuning improves several safety and truthfulness metrics, but `Toxigen` worsens for `9B IT` (Table 6), indicating remaining trade-offs in safety fine-tuning.

- Limited scale exploration
  - Only 2B and ~9B models are released; it is not shown how Griffin-based models behave at much larger scales or with vastly larger training corpora.

- Large vocabulary overhead
  - A 256k vocabulary makes the embedding layers a sizable portion of parameters (Table 1), which may affect memory footprint and distillation/serving strategies, even though recurrent state is compact.

- Reproducibility of RLHF recipe
  - The paper references a “novel RLHF algorithm” but does not detail it here; reproducing instruction‑tuned behavior exactly may require additional documentation.

## 7. Implications and Future Directions
- Field-level impact
  - Demonstrates that non‑Transformer architectures with fixed‑size memory can reach mainstream LM accuracy while decisively outperforming on long-sequence efficiency. This pressures the default assumption that global attention is necessary for strong language modeling at small and mid scales.

- Practical applications
  - Long-form generation and streaming settings (e.g., assistants maintaining long histories), code completion over large files, log analysis, and on-device or edge deployment where memory is constrained. The constant memory footprint makes serving costs more predictable.

- Research opportunities
  - Long-context quality: Benchmark explicitly on long-range reasoning/retrieval tasks to test whether recurrence + local windows suffice, and to tune window size vs. quality trade-offs.
  - Detailed ablations: Quantify the contribution of embedding scaling, RG-LRU regularization choices, and gradient clipping to stability and accuracy.
  - Scaling laws: Explore larger Griffin models and token budgets to characterize efficiency/accuracy trade-offs at scale.
  - Kernel and hardware portability: Develop GPU-optimized recurrence kernels to narrow the gap between TPU and GPU throughput.
  - Safety and RLHF: Investigate why `Toxigen` worsens for `9B IT` and refine reward models/training to improve across all safety metrics.
  - Hybrid designs: Combine recurrence with sparse/global attention or retrieval-augmented mechanisms for tasks that truly need long-distance, non-local dependencies.

Overall, RecurrentGemma shows that Griffin’s fixed-state design can match Transformer accuracy while removing the core bottleneck of KV cache growth, delivering large real-world wins in speed and memory—especially during long generations—without requiring massive compute or model sizes (Figures 1a–1b; Table 2).
