# RecurrentGemma: Moving Past Transformers for Efficient Open Language Models

**ArXiv:** [2404.07839](https://arxiv.org/abs/2404.07839)
**Authors:** Aleksandar Botev, Soham De, Samuel L. Smith, Anushan Fernando, George‑Cristian Muraru, Ruba Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, Léonard Hussenot, Johan Ferret, Sertan Girgin, Olivier Bachem, Alek Andreev, Kathleen Kenealy, Thomas Mesnard, Cassidy Hardin, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Armand Joulin, Noah Fiedel, Evan Senter, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, David Budden, Arnaud Doucet, Sharad Vikram, Adam Paszke, Trevor Gale, Sebastian Borgeaud, Charlie Chen, Andy Brock, Antonia Paterson, Jenny Brennan, Meg Risdal, Raj Gundluru, Nesh Devanathan, Paul Mooney, Nilay Chauhan, Phil Culliton, Luiz Gustavo Martins, Elisa Bandy, David Huntsperger, Glenn Cameron, Arthur Zucker, Tris Warkentin, Ludovic Peran, Minh Giang, Zoubin Ghahramani, Demis Hassabis, Yee Whye Teh, Nando de Frietas
**Institutions:** Google DeepMind

## 🎯 Pitch

RecurrentGemma presents a novel family of language models utilizing linear recurrence combined with local attention, eliminating the need for memory-intensive global transformer attention. This innovation dramatically enhances throughput and efficiency for long-sequence processing, making it ideal for resource-constrained environments and tasks requiring extended generation without compromising on accuracy.

---

## 1. Executive Summary (2-3 sentences)
RecurrentGemma introduces an open family of language models built on the `Griffin` architecture, which replaces global transformer attention with a hybrid of linear recurrence and local attention. The result is a fixed-size model state that enables much faster, memory-efficient inference on long sequences while maintaining competitive accuracy with similarly sized transformer baselines trained on more data (see Table 2 and Figures 1a–1b).

## 2. Context and Motivation
- Problem addressed:
  - Long-sequence inference with transformers is memory-inefficient because it requires storing a growing `KV cache`—the keys and values from past tokens needed for attention—which scales linearly with sequence length.
  - This becomes a bottleneck in resource-constrained deployments (edge devices, limited-GPU memory) and slows generation as sequences get longer.
- Why it matters:
  - Lower memory and higher throughput directly reduce serving costs and unlock applications requiring very long generations (e.g., streaming assistants, long-form content, real-time code generation).
  - The work also explores whether non-transformer architectures can match transformer-level quality at similar parameter counts and less pretraining data.
- Prior approaches and limitations:
  - Local attention in transformers shrinks the cache but typically harms performance for tasks that require longer-range dependencies.
  - Efficient transformer variants (e.g., Multi-Query Attention) reduce memory but still keep an attention cache that grows with sequence length.
- Positioning of this work:
  - RecurrentGemma leverages `Griffin`, which “mixes linear recurrences with local attention” to model sequences without global attention. The key idea is to maintain a fixed-size recurrent state, avoiding a growing cache while preserving locality where it matters (2K-token local window; Table 1).
  - The paper shows comparable accuracy to Gemma transformers despite training on fewer tokens, and significantly better long-sequence throughput (Tables 2, Figures 1a–1b).

## 3. Technical Approach
This section explains what the model is, how it differs from a transformer, and how it is trained and evaluated.

- What is `Griffin`?
  - Griffin is an architecture that combines:
    - Linear recurrences: A per-token update of a fixed-size hidden state. Intuitively, the model “summarizes” all prior context into a fixed-size memory vector that is updated as each new token arrives. This avoids storing all past tokens.
    - Local attention: Attention over a fixed-size window around each position (2048 tokens here; Table 1) to capture recent context and local interactions that benefit from attention’s flexibility.
  - The result is a sequence model without global attention, using constant memory beyond the local window.
  - The paper adopts Griffin as-is, with one small but impactful modification (see below).

- What changes does RecurrentGemma make to Griffin?
  - Input scaling and tied embeddings:
    - It “multiply[s] the input embeddings by a constant equal to the square root of model width.” The input and output embeddings are tied, but the scaling is applied only on the input side (Model architecture section).
    - This mirrors a scaling trick used in Gemma and stabilizes optimization when widths are large.
  - Optimization details for the recurrent layers:
    - “We do not apply weight decay to the parameters of the recurrent (RG-LRU) layers during training.”
    - When backpropagating through the square root operation in the recurrent layers, “we always clip the derivative to a maximum value of 1000 for stability.”
    - These choices aim to maintain stable training dynamics for the recurrent components.

- Model sizes and key hyperparameters (Table 1):
  - `RecurrentGemma-2B`:
    - Total params: 2.68B (2.03B non-embedding + 0.65B embedding), width: 2560, depth: 26, attention heads: 10, local window: 2048, MLP expansion: 3, vocabulary: 256k.
  - `RecurrentGemma-9B`:
    - Total params: 8.58B (7.53B non-embedding + 1.05B embedding), width: 4096, depth: 38, attention heads: 16, local window: 2048, MLP expansion: 3, vocabulary: 256k.
  - Note the large vocabulary (256k) causes embeddings to be a significant fraction of total parameters.

- Why linear recurrence + local attention?
  - Local attention preserves the strengths of attention where it matters most (recent context).
  - Linear recurrence provides a constant-memory summary of the entire history beyond the local window, avoiding global attention’s O(n) memory growth via a KV cache.
  - This design directly targets the memory bottleneck and the throughput drop that transformers experience as sequences grow.

- Training setup (Training details):
  - Pretraining:
    - Sequence length: 8192 tokens.
    - Data: Same sources as Gemma (primarily English web, math, code) with safety filtering and removal of evaluation sets.
    - Tokens seen: both `RecurrentGemma-2B` and `-9B` train on 2T tokens.
    - Two-phase curriculum: train first on a large general mixture, then continue on a smaller, higher-quality dataset.
    - Tokenizer: subset of SentencePiece with 256k vocabulary.
  - Instruction tuning + RLHF (Instruction tuning and RLHF):
    - Follows Gemma’s SFT + a novel RLHF procedure to optimize reward for instruction-following and dialogue.
    - Dialogue format controlled by explicit special tokens; see Table 3 (token names) and Table 4 (example conversation).

- Inference and implementation:
  - Fixed-size state:
    - The recurrent pathway compresses context into a bounded state, so memory does not grow with sequence length beyond the 2K window.
    - This specifically addresses the transformer KV cache problem: “RecurrentGemma compresses input sequences into a fixed-size state without sacrificing performance. This reduces memory use and enables efficient inference on long sequences.”
  - Throughput:
    - Sampling throughput is much higher and does not degrade with sequence length (Figures 1a–1b).
    - Prompt processing is parallelizable and similar to Gemma in speed; the main gains are in autoregressive generation.
  - Software/hardware:
    - Released efficient JAX/Flax code plus a specialized `Pallas` kernel for TPUs; PyTorch reference provided.
    - The throughput figures were captured on TPUv5e (2B) and TPUv4 (9B) using the Flax implementation with the Pallas kernel; “Users should expect lower throughput when using the Pytorch implementation or when using GPUs.”

- A simple mental model:
  - Think of RecurrentGemma as having two “memories” per layer:
    - A small, learnable “long-term memory” (`linear recurrence`) that is updated once per token and whose size does not depend on past sequence length.
    - A “short-term memory” (`local attention`) that can look back over a 2K-token window to resolve fine-grained dependencies.
  - The model mixes these two signals to produce each layer’s output.

## 4. Key Insights and Innovations
- Fixed-size state with competitive accuracy:
  - Innovation: Replace global attention with “linear recurrences + local attention” so the model state does not grow with sequence length.
  - Why it matters: This breaks the memory–length coupling of transformers. The paper shows that both `RecurrentGemma-2B` and `-9B` are “competitive with the Gemma models” on standard benchmarks (Table 2) despite training on fewer tokens (2T vs 3T for Gemma-2B and 6T for Gemma-7B).
  - Significance: Fundamental innovation in modeling long sequences efficiently without giving up quality.

- Practical training stabilizations for recurrent layers:
  - Innovation: Input-embedding scaling by sqrt(width), no weight decay on RG-LRU params, and clipping the derivative through sqrt to 1000.
  - Why it matters: Recurrent components can be sensitive to optimization dynamics. These choices make the hybrid architecture train reliably at scale (Model architecture section).
  - Significance: Incremental but important—turns a promising idea (Griffin) into a stable, open model release.

- Throughput that is robust to sequence length:
  - Innovation: Inference throughput “does not reduce as the sequence length increases,” unlike transformers where throughput falls as the KV cache grows (Figures 1a–1b).
  - Why it matters: For long generations, RecurrentGemma significantly outpaces transformer baselines, with “particularly large (up to two orders of magnitude) improvements over Gemma-7B” in the 9B setting (Figure 1b).
  - Significance: Directly impacts deployment costs and enables arbitrarily long generations within fixed memory.

- Instruction-tuned models with strict dialogue formatting:
  - Innovation: Clear control tokens to enforce turn-taking and structure (Tables 3–4).
  - Why it matters: Predictable formatting supports safer and more controllable instruction following and makes it easier to integrate the model in chat systems.
  - Significance: Incremental but practical; helps align the model’s interactive behavior.

## 5. Experimental Analysis
- Evaluation methodology:
  - Benchmarks (Table 2) span knowledge and reasoning (MMLU, BBH), commonsense (HellaSwag, PIQA, SIQA), QA (TriviaQA, Natural Questions), program synthesis (HumanEval, MBPP), and math (GSM8K, MATH), among others.
  - Safety benchmarks (Table 6) include RealToxicity (lower is better), BOLD, CrowS-Pairs, BBQ, Winogender, TruthfulQA, Winobias, Toxigen.
  - Human evaluation (Table 5) compares instruction-tuned RecurrentGemma to Mistral 7B v0.2 Instruct on separate instruction-following and safety prompt sets.
  - Inference speed (Figures 1a–1b) measures max tokens per second for sampling across generation lengths and for processing prompts across prompt lengths.

- Main quantitative results:
  - Accuracy vs Gemma (Table 2):
    - `RecurrentGemma-2B` vs `Gemma-2B`:
      - Average across tasks: 44.6 vs 45.0.
      - Notable close matches: HellaSwag 71.0 vs 71.4; PIQA 78.5 vs 77.3 (slightly better); GSM8K 13.4 vs 17.7 (worse).
    - `RecurrentGemma-9B` vs `Gemma-7B` (similar total param counts, training data: 2T vs 6T):
      - Average: 56.1 vs 56.9.
      - Some strong spots: TriviaQA 70.5 vs 63.4; Winogrande 73.6 vs 72.3; BBH 55.2 vs 55.1.
  - Human evaluation (Table 5):
    - Safety: Both IT models win ~60% vs Mistral 7B v0.2 Instruct.
      - `2B IT`: 59.8% win rate; `9B IT`: 59.9% win rate.
    - Instruction-following:
      - `2B IT`: 43.7% win rate (competitive given it’s smaller).
      - `9B IT`: 59.3% win rate, outperforming Mistral 7B v0.2 Instruct.
  - Throughput (Figures 1a–1b):
    - Sampling: Throughput for RecurrentGemma “does not reduce as the sequence length increases,” whereas Gemma’s drops with length.
    - Magnitude: The 9B model shows “up to two orders of magnitude” improvement over Gemma-7B for long generations (Figure 1b).
    - Prompt processing: Similar between architectures; “roughly 40K tokens per second for the 2B models and roughly 12K tokens per second for the 9B model.” During sampling, “RecurrentGemma achieves throughput of 6K tokens per second,” with Gemma “substantially slower.”
  - Safety benchmarks (Table 6):
    - Post instruction-tuning, RealToxicity decreases (better): 2B PT 9.8 → IT 7.6; 9B PT 10.3 → IT 8.8.
    - TruthfulQA improves after IT: 2B 35.1 → 42.7; 9B 38.6 → 47.7.
    - Mixed outcomes: On Toxigen, 9B IT is worse than 9B PT (64.5 vs 58.8).

- Do the experiments support the claims?
  - Efficiency: Yes, clearly. The fixed-size state leads to length-robust throughput improvements in sampling, which is the real-world bottleneck (Figures 1a–1b).
  - Accuracy: Also broadly supported. Averages and many task-level scores are comparable to Gemma at the same or similar parameter scales, despite substantially fewer training tokens (Table 2).
  - Alignment and safety: Human evals show strong safety wins and solid instruction-following for `9B IT` (Table 5). Academic safety benchmarks generally improve after IT, with some exceptions (Table 6).
- Notable absences and potential ablations:
  - The paper highlights a single architectural modification vs Griffin and a few optimizer choices, but does not present ablations isolating the impact of each (e.g., effect of input scaling or derivative clipping).
  - No explicit long-context task evaluation (e.g., long-context QA) to quantify quality preservation at very long lengths—though throughput benefits are demonstrated.
  - The paper does not include scaling law analyses across token counts to quantify data efficiency claims beyond the comparison to Gemma’s training tokens.

## 6. Limitations and Trade-offs
- Assumptions and scope:
  - The approach presumes that a mix of local attention (2K window) and a fixed-size recurrent state can capture dependencies sufficiently for general-purpose language tasks. Tasks that truly need unbounded exact attention over distant tokens may still favor transformer-style global attention.
- Expressivity vs memory:
  - The fixed-size state is a design trade-off: constant memory and stable throughput at the possible cost of modeling extremely long-range, precise pairwise interactions that global attention can capture.
- Data and training:
  - Both models are trained on 2T tokens (less than the Gemma baselines), which is a strength for efficiency, but the paper does not report how performance evolves with more data or different mixtures.
  - The specific “novel RLHF algorithm” is not detailed; reproducibility and sensitivity analyses are therefore limited.
- Implementation constraints:
  - Throughput gains reported are on TPUs using a custom Flax `Pallas` kernel. The paper cautions: “Users should expect lower throughput when using the Pytorch implementation or when using GPUs.”
- Safety and robustness:
  - Safety results are mixed on some tests (e.g., 9B IT Toxigen worsens vs PT), indicating remaining risks and a need for use-case-specific safety testing (Responsible Deployment section).
- Architecture details:
  - The paper defers most architectural specifics to the Griffin paper. Some readers may want more in-paper detail (e.g., exact recurrence equations, gating structure) to fully reason about capabilities and limits.
- Parameter budget:
  - The 256k vocabulary causes a large embedding matrix (Table 1), which increases total parameters and memory footprint even if inference state is small.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that non-transformer sequence models can be competitive at scale while offering dramatic efficiency advantages for long generations. This pressures the default choice of global attention for many deployment scenarios.
  - For on-device or cost-sensitive serving, RecurrentGemma’s fixed-size state offers a compelling alternative to transformers.
- Practical applications:
  - Long-form generation (novels, documentation), always-on assistants (streaming conversation or coding), interactive agents with long-running sessions, and batch high-throughput generation workloads where serving cost is paramount.
  - Scenarios where prompt processing is short relative to generated output—since sampling dominates wall-clock time, the throughput gains are most impactful (Figures 1a–1b).
- Follow-up research directions:
  - Long-context quality: Evaluate on tasks designed to stress extremely long-range dependencies and retrieval, including document-level QA and multi-document reasoning.
  - Architecture ablations: Quantify the contribution of local window size, recurrent state width, gating strategies, and the training stabilizations (input scaling, derivative clipping, no weight decay for RG-LRU).
  - RLHF transparency and robustness: Publicly detail the “novel RLHF algorithm,” assess robustness to adversarial prompts, and explore policy constraints that improve outcomes on mixed safety benchmarks (Table 6).
  - Cross-lingual and domain specialization: The dataset is primarily English; multilingual and domain-focused variants would test generality.
  - Tool use and retrieval: Explore how fixed-state recurrent models interact with external memory (retrieval) and tools, potentially reducing the required local window while maintaining accuracy.
  - Software portability: Optimize GPU kernels and PyTorch paths to close the performance gap with the TPU-specific Pallas implementation.

Quoted highlights for quick reference:
- “RecurrentGemma compresses input sequences into a fixed-size state without sacrificing performance. This reduces memory use and enables efficient inference on long sequences.”
- “We make only a single modification to the Griffin architecture… multiply the input embeddings by… the square root of model width… [and] do not apply weight decay to the parameters of the recurrent (RG-LRU) layers… [and] clip the derivative to a maximum value of 1000 for stability.”
- Training tokens: “both RecurrentGemma-2B and RecurrentGemma-9B [trained] on 2T tokens,” vs Gemma-2B at 3T and Gemma-7B at 6T (Table 2 caption).
- Throughput: “RecurrentGemma achieves higher sampling throughput… [and] does not reduce as the sequence length increases,” with “particularly large (up to two orders of magnitude) improvements over Gemma-7B” (Figures 1a–1b).
- Prompt vs sampling speeds: “roughly 40K tokens per second for the 2B models and roughly 12K tokens per second for the 9B model” on prompts; during sampling, “RecurrentGemma achieves throughput of 6K tokens per second,” with Gemma “substantially slower.”
- Human eval vs Mistral 7B v0.2 Instruct: Instruction following win rate 59.3% for `RecurrentGemma-9B IT`; Safety win rates ~60% for both IT models (Table 5).
