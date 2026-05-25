# MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

**ArXiv:** [2402.14905](https://arxiv.org/abs/2402.14905)

## 🎯 Pitch

MobileLLM challenges the prevailing belief that only model size and data scale matter by demonstrating that careful architectural choices—such as deeper and thinner networks, strategic weight sharing, and grouped-query attention—enable sub-billion-parameter language models to achieve state-of-the-art accuracy while remaining fast and energy-efficient for mobile devices. By specifically designing and benchmarking models to fit within the tight memory and computational limits of smartphones, MobileLLM enables high-quality on-device language understanding, reducing reliance on the cloud for lower latency, improved privacy, and greatly improved accessibility.

---

## 1. Executive Summary
MobileLLM designs and trains sub‑billion‑parameter language models that run efficiently on mobile devices by prioritizing architecture choices and weight reuse over sheer parameter count. It delivers new 125M and 350M models—plus a layer‑sharing variant (`MobileLLM‑LS`)—that set state‑of‑the‑art accuracy among peers while keeping memory and latency compatible with phone hardware.

## 2. Context and Motivation
- Problem addressed
  - How to build high‑quality language models that can run locally on phones without relying on the cloud.
  - On-device memory is tight: DRAM is typically 6–12 GB and a single app should stay well below that budget (Figure 2). LLM inference is also energy‑intensive; a 7B‑parameter model can consume ~0.7 J/token, making sustained mobile use impractical (Section 1).
- Why this matters
  - Practical: On‑device inference reduces latency, bandwidth, cloud costs, and privacy risks. Section 1 argues that if people use LLMs for ~5% of daily time, purely cloud‑based serving at GPT‑4‑scale would need ~10^8 H100 GPUs—economically and environmentally daunting (Appendix I).
  - UX: Measured decoding speed on phones is far higher for smaller models; a 125M model can reach ~50 tokens/s, whereas a 7B model runs at ~3–6 tokens/s in an iPhone app (Section 1, footnote 5).
- Where prior work falls short
  - Prior sub‑billion models (e.g., OPT‑125M/350M, GPT‑Neo‑125M, Pythia‑160M/410M, RWKV‑169M/430M, BLOOM‑560M) were not architected specifically under strict on‑device memory/latency constraints or did not prioritize architectural choices for tiny models (Tables 3–4).
  - A common belief from scaling laws is that architecture matters little once parameter count and data are fixed (Section 2.2.2). This paper contests that belief at the sub‑billion scale.
- Positioning
  - MobileLLM focuses on architectural and weight‑reuse choices tailored to small models: deep‑and‑thin networks, input/output embedding sharing, grouped‑query attention, and a new block‑wise layer‑sharing scheme. The result is a family of models that outperform prior sub‑billion baselines on reasoning, QA/RC, chat, and an API‑calling task (Tables 3–6).

## 3. Technical Approach
The work proceeds in two stages: build a strong baseline under tight parameter budgets, then add a new layer‑sharing method that preserves model size but improves accuracy and on‑device execution locality.

1) Build a compact yet strong baseline (“MobileLLM,” Section 2.2)
- Deep‑and‑thin Transformer
  - Depth vs width search: for a fixed parameter budget, more layers with smaller hidden sizes perform better than fewer, wider layers on sub‑billion models (Figure 4; Tables 11–12). Example: around 125M parameters, 30 or 42 layers outperform 12 layers across zero‑shot reasoning, QA (TriviaQA), and reading comprehension (RACE).
  - Intuition: extra depth stacks more non‑linear transformations, improving representation power for abstract reasoning, while keeping per‑layer dimensions small to fit memory/compute budgets.
- `SwiGLU` feed‑forward networks (FFN) instead of ReLU FFN
  - `SwiGLU` is a gated activation that improves gradient flow and expressivity in FFNs. Switching to `SwiGLU` yields a +1.3 average accuracy point at 125M and 350M (Table 10, “+ SwiGLU in FFN”).
- Input–output `embedding sharing` (Section 2.2.3)
  - Definition: reuse the same embedding matrix to map tokens to vectors (input) and vectors to logits over the vocabulary (output). In a 32k‑vocab with 512‑dim embeddings, sharing saves ~16M parameters—> >10% at 125M scale (Table 1).
  - How it helps: saved parameters can be reallocated to add layers (“↑ depth”), recouping or improving accuracy at similar total size (Table 1: 30‑layer 125M model with emb‑share + 2 extra layers reaches a higher average than the non‑shared 135M model).
- `Grouped‑Query Attention (GQA)` (Section 2.2.4)
  - Definition: use fewer key/value heads than query heads and reuse (repeat) each KV head across multiple Q heads. If Q heads = 16 and KV heads = 4, each KV head serves 4 Q heads.
  - Why: reduces KV parameter redundancy and KV‑cache memory at inference. Ablations show best performance near 16 Q heads, with KV heads reduced to 4 causing negligible loss (125M) or ~0.2pt drop (350M) while saving size (~10%) (Figure 5; Table 13).
- Baseline architecture choices
  - 125M: 30 layers, 576 embedding dim, 9 heads, 3 KV heads (Table 9 in Appendix A).
  - 350M: 32 layers, 960 dim, 15 heads, 5 KV heads.
- Training setup (Sections 2.1 and 3.1)
  - Hardware: 32× A100 GPUs, batch size 32/GPU.
  - Data/steps: exploration on 0.25T tokens for 120k iters; final models trained 480k iters on ~1T tokens.
  - Optimizer/schedule: Adam, weight decay 0.1, initial LR 2e‑3 with cosine decay.

2) Add `immediate block‑wise layer sharing` (“MobileLLM‑LS,” Section 2.3)
- Definition: make adjacent Transformer blocks share weights and compute the same block twice in sequence. No new parameters; the block is reused immediately while its weights are still hot in cache (Figure 6b).
- Why this is different from other sharing layouts (Figure 6)
  - “Repeat‑all‑over” and “reverse” sharing slightly improve accuracy but do not respect mobile cache realities; a shared block may be evicted before reuse, forcing extra DRAM traffic (Table 2; Figure 2).
  - Immediate block‑wise sharing maximizes data locality: SRAM/L3 caches on phones are ~8–32MB (Figure 2, device table), typically fitting one block’s weights. Reusing the block immediately avoids new weight fetches from DRAM.
- Empirical effect
  - Accuracy: modest improvements over the non‑shared baseline (Table 2).
  - Latency on iPhone 13 (ExecuTorch + MPS): almost no overhead vs. non‑shared baseline of the same parameter size, while naive doubling of layers without sharing is much slower (Table 7).

3) Implementation notes
- Quantization compatibility
  - Post‑training quantization (PTQ) to 8‑bit weights and activations (`W8A8`) yields <0.5 average accuracy drop and works with layer sharing (Figure 7; Table 15).
- Distillation
  - Knowledge distillation from LLaMA‑v2‑7B via cross‑entropy on logits (Equation 1) slowed training 2.6–3.2× and gave no accuracy gain; the authors therefore train with standard label supervision (Table 16).

## 4. Key Insights and Innovations
- Depth beats width for tiny LLMs (fundamental insight)
  - Finding: at fixed size, more layers with smaller dimensions consistently outperform shallower, wider models on eight zero‑shot commonsense tasks (Figure 4a–b), TriviaQA (Figure 4c–d), and RACE (Figure 4e–f). Detailed sweeps are in Tables 11–12.
  - Significance: challenges the common reading of scaling laws that architecture matters little once size/data are fixed (Section 2.2.2). For sub‑billion models, design choices are decisive.
- Weight reuse across the stack (architectural economy)
  - Embedding sharing (Table 1) and GQA (Figure 5; Table 13) reclaim parameters and KV‑cache budget without sacrificing accuracy; savings are reinvested into more layers. This reframes “where to spend parameters” for small models.
- Immediate block‑wise layer sharing (systems‑aware innovation)
  - Novelty: a sharing pattern matched to mobile memory hierarchies—reuse a block immediately to exploit cache locality (Figure 6b).
  - Evidence: accuracy improves over non‑shared baselines (Table 2) and latency overhead on iPhone 13 is minimal compared to a naïve 2×‑deeper model (Table 7: +2.6% execute time vs +86% without sharing).
- Demonstrated on‑device viability and capability (applied contribution)
  - Models not only score higher on standard reasoning/QA benchmarks but also show strong chat performance and near‑7B‑level exact‑match for API intent/structure prediction (Tables 5–6), indicating fitness for common phone assistants.

## 5. Experimental Analysis
- Evaluation methodology
  - Zero‑shot commonsense: ARC‑easy/challenge, BoolQ, PIQA, SIQA, HellaSwag, OBQA, WinoGrande (Section 3.2; Table 3).
  - QA and reading comprehension: TriviaQA (1/5/64‑shot F1) and RACE (middle/high accuracy) (Table 4).
  - Chat: AlpacaEval (GPT‑4 judge) and MT‑Bench (GPT‑4 judge; multi‑turn) with identical fine‑tuning pipelines across baselines (Section 3.3.1; Table 5).
  - API calling: synthetic dataset, 5k train/2.5k test, ~8 turns per dialog; measures `EMintent` and `EMstructure` for the JSON API spec plus ROUGE‑1/L for agent replies (Section 3.3.2; Table 6).
  - On‑device latency: iPhone 13 via ExecuTorch + Metal Performance Shaders (MPS) (Section 3.6; Table 7).
  - Ablations: FFN activation, depth/width, embedding sharing, heads/KV‑heads, sharing layout, repetition counts (Tables 1–2, 10–14; Figures 4–5).
- Main quantitative results
  - Zero‑shot commonsense (Table 3)
    - 125M:
      - `MobileLLM‑125M`: average 46.3 vs prior 125–170M models: 42.5–43.6.
      - `MobileLLM‑LS‑125M`: average 47.0.  
        > From Table 3: “MobileLLM‑LS‑125M … Avg. 47.0,” beating OPT‑125M (42.6), GPT‑Neo‑125M (42.9), Pythia‑160M (42.5), and RWKV‑169M (43.6).
    - 350M:
      - `MobileLLM‑350M`: average 51.3; `MobileLLM‑LS‑350M`: 52.1.  
        > Table 3 shows `MobileLLM‑LS‑350M` outperforms Pythia‑410M (46.6), RWKV‑430M (47.0), BLOOM‑560M (44.2), and OPT‑350M (43.9) by 4–8 points.
  - QA and reading comprehension (Table 4)
    - `MobileLLM‑125M`: TriviaQA 1/5/64‑shot F1 = 13.9/14.3/12.5; RACE middle/high = 39.7/28.9.
    - `MobileLLM‑350M`: 22.0/23.9/24.2 on TriviaQA; RACE middle/high = 45.6/33.8.  
      > Table 4: the 350M model is ≈10 F1 points higher than other 350–590M models on TriviaQA and clearly better on RACE.
  - Chat (Table 5)
    - `MobileLLM‑LS‑350M`: MT‑Bench 3.16 and AlpacaEval 48.20% win vs text‑davinci‑001 baseline (GPT‑3).  
      > Table 5: within <1B models, MobileLLM variants substantially exceed OPT‑350M (1.37/6.80) and BLOOM‑560M (1.73/10.29).
  - API calling (Table 6)
    - `MobileLLM‑350M`: `EMintent` 65.3 and `EMstructure` 48.8 vs LLaMA‑v2 7B at 62.8 and 50.9.  
      > Table 6: intent classification is slightly higher than 7B; structure exact‑match is close; ROUGE is lower (46.8/44.6 vs 56.5/54.3), which matters less for API correctness.
  - Quantization (Figure 7; Table 15)
    - `W8A8` PTQ yields ≤0.5 average drop across 125M/350M and with/without layer sharing.  
      > Table 15: e.g., `MobileLLM‑350M` BF16 = 49.9 avg; W8A8 = 49.9.
  - On‑device latency (Table 7)
    - For 125M on iPhone 13: load/init/execute times are 39.2/1361.7/15.6 ms for baseline vs 43.6/1388.2/16.0 ms for `LS` (≈2–3% overhead).  
      > Table 7 also contrasts a naïve 60‑layer (non‑shared) model: 68.6/3347.7/29.0 ms—much slower to load/init and execute.
- Ablations and what they teach
  - FFN activation: `SwiGLU` consistently improves accuracy over vanilla FFN (Table 10).
  - Depth/width: clear monotonic gains up to ~30 layers for both sizes; very shallow models (<10 layers) are weak on reasoning and comprehension (Tables 11–12).
  - Embedding sharing: saves ~11.8% parameters at 125M with negligible average drop, recovered by adding 2 layers (Table 1).
  - GQA: 16 Q heads with 4 KV heads balances accuracy and memory; validated at both scales (Figure 5; Table 13).
  - Layer‑sharing layout: “repeat‑all‑over” may edge out “immediate” in average accuracy, but immediate sharing is chosen for cache locality on phones (Table 2 and Section 2.3).
  - Repetition times: doubling layers with sharing helps (+0.4–0.6 avg) but further ×3 or ×4 gives diminishing or inconsistent returns (Table 14).
- Do experiments support the claims?
  - The breadth of tasks (commonsense, QA/RC, chat, API) and the extensive ablations strongly support the central claims: depth matters, weight reuse works, and block‑wise sharing achieves the desired accuracy/latency balance.
  - One caveat: chat metrics use GPT‑4 as judge (Table 5), which can introduce evaluation bias; nevertheless, comparisons were run under identical settings.

## 6. Limitations and Trade-offs
- Training cost and data
  - Final models are trained for 480k iterations on ~1T tokens (Section 3.1). While inference is light, pretraining is still compute‑heavy; the data composition is not detailed, so domain bias and data quality control are opaque.
- Language and task coverage
  - Evaluations focus on English benchmarks. Long‑context tasks, tool‑use beyond the provided API dataset, and multilingual settings are not studied.
- Layer sharing trade‑offs
  - Sharing increases compute (the block is applied twice), though execution remains fast because memory traffic is the bottleneck on phones (Table 7). On architectures where compute is the bottleneck, sharing could be less favorable.
  - Sharing may constrain representational diversity across adjacent layers. The paper’s ablations show gains at ×2 repetition (Table 14) but not monotonic improvements beyond that.
- Embedding sharing constraints
  - Tying input and output embeddings can reduce flexibility in specialized finetuning where decoupling could help. The paper offsets this by increasing depth, but specific finetuning trade‑offs aren’t explored.
- GQA head reduction
  - Fewer KV heads reduce cache size, but potential impacts on tasks requiring very fine‑grained multi‑head representations or extremely long context attention are not evaluated.
- Chat evaluation
  - GPT‑4‑judged benchmarks are helpful but imperfect proxies for real user satisfaction and safety. Human evaluation and safety audits are out of scope.

## 7. Implications and Future Directions
- How this changes the landscape
  - For sub‑billion LLMs, architecture matters decisively. The recipe—deep‑and‑thin, embedding sharing, GQA, and immediate block‑wise sharing—redefines best practice for models meant to live on phones.
  - The results close the gap between tiny models and multi‑billion models on practical tasks like API calling:  
    > Table 6 shows `MobileLLM‑350M` matching or surpassing 7B on intent prediction and approaching it on structure.
- Practical applications
  - On‑device assistants for messaging, reminders, and app control; privacy‑sensitive chat; offline query routing; low‑latency UI helpers. The latency study (Table 7) and quantization results (Table 15) indicate deployability on current phones.
- Follow‑up research
  - Hardware‑aware architecture search at the sub‑billion scale to generalize the depth‑first principle across devices and NPUs.
  - Dynamic or conditional layer sharing: choose when to reuse vs specialize blocks based on input difficulty to balance accuracy and speed.
  - Long‑context and memory‑augmented variants under the same parameter budgets; evaluate with retrieval and tool‑use over larger API/task suites.
  - Safety/robustness and multilingual training pipelines at these sizes.
  - Further integration with sparsity/pruning and mixed‑precision techniques; the paper already demonstrates compatibility with `W8A8` PTQ (Figure 7; Table 15).

Overall, MobileLLM demonstrates that thoughtful architectural design and weight reuse can outperform simply scaling parameters for small models. With concrete, system‑aware choices and thorough ablations, it delivers a practical path to high‑quality, on‑device LLMs.
