# Pretraining Large Language Models with NVFP4

**ArXiv:** [2509.25149](https://arxiv.org/abs/2509.25149)
**Authors:** NVIDIA, Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, Evan Briones, Ian Buck, Bryan Catanzaro, Jinhang Choi, Mike Chrzanowski, Eric Chung, Victor Cui, Steve Dai, Bita Darvish Rouhani, Carlo del Mundo, Deena Donia, Burc Eryilmaz, Henry Estela, Abhinav Goel, Oleg Goncharov, Yugi Guvvala, Robert Hesse, Russell Hewett, Herbert Hum, Ujval Kapasi, Brucek Khailany, Mikail Khona, Nick Knight, Alex Kondratenko, Ronny Krashinsky, Ben Lanir, Simon Layton, Michael Lightstone, Daniel Lo, Paulius Micikevicius, Asit Mishra, Tim Moon, Deepak Narayanan, Chao Ni, Abhijit Paithankar, Satish Pasumarthi, Ankit Patel, Mostofa Patwary, Ashwin Poojary, Gargi Prasad, Sweta Priyadarshi, Yigong Qin, Xiaowei Ren, Oleg Rybakov, Charbel Sakr, Sanjeev Satheesh, Stas Sergienko, Pasha Shamis, Kirthi Shankar, Nishant Sharma, Mohammad Shoeybi, Michael Siu, Misha Smelyanskiy, Darko Stosic, Dusan Stosic, Bor‑Yiing Su, Frank Sun, Nima Tajbakhsh, Shelby Thomas, Przemek Tredak, Evgeny Tsykunov, Gandhi Vaithilingam, Aditya Vavre, Rangharajan Venkatesan, Roger Waleffe, Qiyu Wan, Hexin Wang, Mengdi Wang, Lizzie Wei, Hao Wu, Evan Wu, Keith Wyss, Ning Xu, Jinze Xue, Charlene Yang, Yujia Zhai, Ruoxi Zhang, Jingyang Zhu, Zhongbo Zhu
**Institutions:** NVIDIA

## 1. Executive Summary (2-3 sentences)
This paper shows that you can pretrain large language models mostly in 4‑bit floating point using a new format, `NVFP4`, without losing accuracy compared to standard `FP8` training—even for a strong 12B-parameter model trained for 10 trillion tokens. The core move is a carefully engineered combination of quantization tricks: two-level microscaling with smaller blocks, “make-the-forward-and-backward-see-the-same-weights” 2D scaling, Random Hadamard Transforms to tame outliers, and stochastic rounding for unbiased gradients (Sections 2–4; Figures 1, 4–5).

## 2. Context and Motivation
- The gap this paper targets
  - Training frontier LLMs takes tens to hundreds of yottaflops. That’s a massive compute, time, and energy bill. Reducing precision is the most direct way to get more math throughput and memory savings.
  - `FP8` training is already a thing. Going from 8-bit to 4-bit would potentially double or triple GEMM throughput on NVIDIA Blackwell GPUs and cut operand memory in half (Table 1), but 4-bit training is notoriously unstable: quantization can wreck gradients, saturate outliers, and even break the chain rule if the forward/backward paths don’t see the “same” function.
- Why this matters
  - Real-world: If `FP4` works for pretraining, we get cheaper, faster training for future frontier models—potentially the difference between feasible and unfeasible runs at trillion-token scales.
  - Theoretical/numerical: It’s a test of whether careful quantization design can preserve optimization dynamics over extremely long horizons (10T tokens), which is where small numerical issues compound.
- What existed before and where it fell short
  - `MXFP4` (microscaling FP4) exists and works for training in some settings, but its block design and power-of-two-only scales force trade-offs that reduce effective dynamic range and can waste FP4 value bins (Appendix B.4). That leads to worse convergence and more tokens needed to reach the same loss (Section 5; Figure 6).
  - Various recipes for low-precision (FP8, some FP4 variants) tackle parts of the problem (outliers, rounding bias, scaling strategy), but do not demonstrate stable multi-trillion-token pretraining at this scale with near-baseline accuracy.
- How this paper positions itself
  - Introduces `NVFP4`, a new FP4 microscaling format with smaller blocks (16 vs 32), fractional `E4M3` block scales, plus a tensor-level `FP32` scale—together providing tighter and more accurate scaling (Section 2; Figure 1).
  - Pairs the format with a training methodology that systematically addresses the known failure modes of FP4 training: outliers (Random Hadamard Transforms), bias (stochastic rounding on gradients), and forward/backward mismatch (2D weight scaling) while keeping a small fraction of sensitive layers in higher precision (Section 4; Figure 4).
  - Validates with a 12B model on 10T tokens, showing losses and downstream accuracy track an `FP8` baseline (Section 3; Figures 2–3; Table 2). Also compares against `MXFP4` and shows better data-efficiency (Section 5; Figure 6).

## 3. Technical Approach
Here’s how the method works end-to-end. I’ll start with the `NVFP4` format (what exactly is stored and why), then the training recipe (what’s quantized, how, and where).

- What is `NVFP4`?
  - It’s a 4‑bit floating format built on “microscaling”: you store values in small blocks; each block shares a higher-precision scale factor so 4‑bit values can represent a wider dynamic range (Section 2).
  - Three design choices differentiate `NVFP4` from `MXFP4` (Section 2; Figure 1):
    1) Smaller blocks: `16` elements per block (vs `32` in `MXFP4`). Smaller blocks mean each block’s values are more locally similar, so fewer elements get crushed to zero or saturate.
    2) Fractional block scales: `E4M3` format for block scale factors (vs `UE8M0` power-of-two-only in `MXFP4`). Having mantissa bits lets you map the block’s max value (“`amax`”) closer to the FP4 max representable value, improving usage of FP4 bins.
    3) Two-level scaling: a per-tensor `FP32` scale plus the per-block `E4M3` scale. The tensor-level scale “preconditions” the entire tensor so block scales stay within the representable range of `E4M3` (Section 2; Appendix B).

- How quantization actually happens (Appendix B; equations (1)–(5))
  - Plain-language summary:
    - Compute a global tensor-scale so the largest value across the tensor would be representable after block scaling (Appendix B.1; equation (1)).
    - For each 16-element block, compute a decode-scale mapping the block’s `amax` to the FP4 max representable, then quantize that block scale to `E4M3` (Appendix B.2; equations (2)–(3)).
    - Scale each value by the inverted per-block encode scale and quantize to FP4 (Appendix B.3; equation (4)).
    - During GEMM, Tensor Cores multiply partial dot products by the stored per-block scales and then by the per-tensor scale to recover the original magnitudes (equation (5)).
  - Why this beats `MXFP4`:
    - `MXFP4` uses power-of-two-only block scales, so after rounding up to avoid saturation you can “waste” top FP4 bins (±4, ±6) and lose almost a full “binade” of dynamic range—i.e., a power-of-two interval (Appendix B.4). `NVFP4`’s fractional `E4M3` block scale and the extra tensor-scale sidestep that.

- Hardware support and performance potential
  - NVIDIA Blackwell Tensor Cores natively support `NVFP4` GEMMs with per-block scale factors and do accumulation in `FP32` (Section 2; Table 1). They also natively support stochastic rounding.
  - Throughput: FP4 GEMMs deliver 2× (GB200) or 3× (GB300) the math throughput of FP8, and operands use ~½ the memory vs FP8 (Table 1). Caveat: end-to-end speedups also depend on overheads (transforms, scaling passes), which this report doesn’t measure (Section 3 end; Section 6).

- The training recipe (Section 4; Figure 5 shows the flow)
  - What gets quantized to `NVFP4`:
    - Most linear layers’ GEMMs: forward (`Fprop`), activation gradients (`Dgrad`), weight gradients (`Wgrad`). Inputs to each GEMM are quantized to `NVFP4` (Figure 5).
  - What stays high precision:
    - “Sensitive” linear layers: roughly 15% of the network, mostly at the end, remain `BF16` or `MXFP8` (Section 4.1). For the 12B experiment they keep the first 2 and last 8 blocks in `BF16` (16% of linear layers).
    - Non-linear and attention parts—embeddings, output head, norms, activation functions, softmax, `QK` and `SV` attention GEMMs—stay `BF16`/`FP32` (Section 4.1).
    - Optimizer states and master weights remain in `FP32`; tensor-parallel reductions happen in `BF16` (Section 4.1).
  - Two consistency tricks to protect optimization:
    1) 2D block scaling for weights (Section 4.3)
       - Problem: if you scale along the dot-product dimension, the backward pass transposes tensors, so the same weight gets quantized differently in forward vs backward. That means gradients don’t correspond to the function actually executed—violating the chain rule (Section 4.3, paragraph 1, with the formulation y_fprop = w_fprop x vs y_bprop = w_bprop x).
       - Fix: quantize weights using `16×16` blocks (2D), so the same 2D region of the weight matrix gets the same scaling in both directions. Activations and gradients keep finer `1×16` blocks for better fidelity (Section 4.3).
       - Evidence: Removing 2D weight scaling worsens loss on the 12B model (Figure 4). A dedicated study shows inconsistent weight quantization hurts more than inconsistent activations (Appendix E.5; Figure 14).
    2) Random Hadamard Transforms (RHT) only on `Wgrad` inputs (Section 4.2; Appendix C–E)
       - What: Apply an orthogonal transform (Hadamard + random sign flips) to redistribute outliers across entries so the block’s dynamic range looks more Gaussian and less spiky—easier for FP4 to represent.
       - Why only on `Wgrad`: Applying transforms along the dot-product dimension also creates forward/backward inconsistency if you transform weights, so they don’t transform weights at all. Empirically, `Fprop`/`Dgrad` didn’t benefit at this scale; `Wgrad` did (Section 4.2; Figure 4; Appendix E.4.1, Figure 11).
       - Details: Use `16×16` tiles; larger sizes slightly help but cost more; too small (e.g., `4×4`) can hurt at 12B scale (Appendix E.4.2; Figure 12). A single fixed random sign vector is enough; removing randomness hurts at scale (Appendix E.4.3; Figure 13).
  - Unbiased gradients via stochastic rounding (SR) (Section 4.4)
    - Deterministic rounding in FP4 biases gradients (values underflow to zero or saturate repeatedly in a consistent direction). SR rounds to neighbors probabilistically to remove bias.
    - Where to apply: on gradients only. Using SR on activations or weights increases quantization noise and caused divergence in ablations (Section 4.4; Figure 4; Appendix E.3; Figure 10).
  - Minimal higher-precision fallback at the end
    - If you need to close the last ~1% loss gap, switch precision late in training (e.g., from `NVFP4` to `BF16` around the start of LR decay). Switching “forward-only” recovers most loss with minimal overhead (Appendix D; Figure 7).

## 4. Key Insights and Innovations
- Two-level, small-block microscaling with fractional block scales is the right shape for FP4 training
  - What’s new: `NVFP4` moves to `16`-element blocks and uses `E4M3` block scales plus a tensor-level `FP32` scale (Section 2; Figure 1).
  - Why it matters: It preserves more FP4 dynamic range and reduces zeros/saturations relative to `MXFP4`’s power-of-two scales, which in turn improves convergence and data efficiency (Section 5; Figure 6; Appendix B.4). This is a foundational improvement, not just incremental.
- Quantize weights with 2D blocks to avoid chain-rule breakage
  - What’s new: Explicitly framing forward/backward quantization inconsistency as “you’re differentiating a different function” and fixing it with `16×16` weight scaling (Section 4.3).
  - Why it matters: This is a conceptual and practical insight—the core of stable training. Without it, the 12B model performs worse (Figure 4) and a 1.2B study shows inconsistent weights are notably damaging (Appendix E.5; Figure 14).
- Outlier control with Random Hadamard Transforms—but only where it helps
  - What’s new: Constraining RHT to `Wgrad` inputs to get the benefit (redistribute outliers) without the cost (forward/backward inconsistency and extra quantization error) (Section 4.2, Appendix E.4.1).
  - Why it matters: A carefully scoped use of RHT stabilizes FP4 at scale. This is a surgical use of an increasingly common idea in quantization (Appendix C; equation (6) mechanics).
- Unbiased gradients via targeted stochastic rounding
  - What’s new: Apply SR only to gradients; use round-to-nearest-even elsewhere (Section 4.4; Figure 4; Appendix E.3; Figure 10).
  - Why it matters: It neutralizes the biggest source of bias without flooding the forward pass with extra quantization noise. This is a practical training recipe point, validated at scale.
- First sustained 4-bit pretraining at multi-trillion-token scale
  - What’s new: A 12B model trained on 10T tokens with FP4-quantized linear layers, tracking FP8 loss and matching downstream accuracy (Section 3; Figures 2–3; Table 2).
  - Why it matters: It’s a strong empirical threshold. It shows the method holds up over long horizons, not just short runs.

## 5. Experimental Analysis
- Setups, datasets, and baselines
  - Main experiment: 12B hybrid Mamba–Transformer (Nemotron-H family) trained for 10T tokens, sequence length 8192, WSD schedule (constant LR 80% then decay) (Section 3; Appendix A.1 for architecture and dataset blend).
    - NVFP4 setup: method from Section 4—most linear GEMMs in FP4; first 2 and last 8 blocks `BF16`; weights `2D` scaling; RHT on `Wgrad` with `16×16`; SR on gradients.
    - Baseline: `FP8` training per NVIDIA/DeepSeek recipes (Appendix A.1).
  - Format comparison: 8B hybrid model trained 1T tokens; compare `NVFP4` vs `MXFP4`, baseline `BF16` (Section 5; Appendix A.2).
- Metrics and evaluation
  - Loss curves (training/validation), and a suite of downstream tasks evaluated in `BF16` inference: `MMLU`, `MMLU-Pro 5-shot`, `AGIEval English CoT`, `GSM8k CoT`, `MATH`, multilingual (`Global MMLU`, `MGSM`), coding (`HumanEval+`, `MBPP+`), and commonsense (`HellaSwag`, `ARC-C`, `PIQA`, `Winogrande`, `OpenBookQA`) (Section 3; Table 2; Figure 3).
- Main quantitative results
  - Loss tracking: In the stable LR phase, NVFP4’s relative loss error vs FP8 stays under 1%; it increases slightly above ~1.5% near the end during LR decay (Figure 2). The small change at 8T tokens corresponds to LR decay; a bump at 9T tokens comes from a dataset blend switch (Figure 2; Appendix A.1).
  - Downstream accuracy parity:
    - Quote (Table 2): “MMLU‑Pro 5‑shot: 62.62 (FP8) vs 62.58 (NVFP4).”
    - Other tasks are largely tied or even favor NVFP4 (e.g., `GSM8k CoT` 92.27 vs 89.08; `MGSM` 85.53 vs 81.87), with coding slightly worse (e.g., `MBPP+` 55.91 vs 59.11); the paper notes coding fluctuations may be eval noise near the final checkpoint (Section 3; Table 2; Figure 3).
  - Ablations confirm each technique matters (Figure 4):
    - Removing SR, RHT, or 2D weight scaling worsens loss; leaving fewer end-of-network blocks in BF16 also worsens loss at 12B scale.
  - Format comparison—NVFP4 vs MXFP4 (Section 5; Figure 6):
    - On an 8B model, NVFP4’s loss is better: relative error ~1.5% vs ~2.5% for MXFP4 (Figure 6a).
    - To match NVFP4, MXFP4 needs 36% more tokens (1.36T vs 1T) (Figure 6b). That’s a big training-time tax.
  - Late-phase precision switch recovers loss (Appendix D; Figure 7):
    - Switching from NVFP4 to BF16 at ~8.2T tokens closes the gap to FP8; switching very late (~10T) helps but can’t fully recover (likely because LR is too low).
    - Switching the forward pass only recovers most of the gap; switching backward only doesn’t help much (Figure 7).
- Do the experiments support the claims?
  - For the central claims—“FP4 pretraining can match FP8 accuracy at scale” and “NVFP4 is more data-efficient than MXFP4”—yes. The 12B/10T run is a strong signal (Figures 2–3; Table 2), and the 8B comparison isolates the format effect (Figure 6).
  - The ablations are thoughtful and target the hypothesized failure modes (bias, outliers, chain-rule mismatch). They start from a halfway-trained 12B model for cost reasons (Figure 4) and also run full 1.2B studies for broader coverage (Appendix E).
- Checks, caveats, and failure modes
  - They do not measure actual wall-clock speedups; only numerical/accuracy results (Section 3 end; Section 6). Some chosen techniques (global amax pass, RHT) add overhead.
  - Coding tasks dip slightly; authors suggest checkpoint noise (Section 3; Table 2), but it’s also plausible code modeling is more sensitive to forward quantization noise.
  - Attention paths remain high precision; the results thus demonstrate “mostly-FP4 linear layers,” not an entirely FP4 network (Section 4.1).
  - The 2D scaling and RHT choices are tuned for these architectures; larger models or different blocks (e.g., pure Transformers, MoE) might need re-tuning (Section 6).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach assumes access to NVIDIA Blackwell GPUs with `NVFP4` Tensor Core support and stochastic rounding instructions (Section 2; Table 1). Portability to other hardware isn’t shown.
  - The dataset blend and schedule are specific (Appendix A.1); performance under very different data regimes isn’t tested.
- What’s not addressed
  - End-to-end runtime efficiency: no throughput/latency or energy numbers. Even if FP4 GEMMs are 2–3× faster than FP8 (Table 1), overhead from extra passes (global amax), transforms, and scale packing could reduce net speedups (Section 6 notes this is out of scope).
  - Full FP4 everywhere: attention GEMMs, norms, and non-linearities remain high precision. Also ~15% of linear layers (mostly final blocks) stay `BF16` (Section 4.1). “FP4 all the way” is not solved here.
- Numerical trade-offs
  - Two-level scaling with a global amax requires an extra memory pass to compute the tensor amax (Appendix B.1). The paper hints you could compute scales at smaller granularity to avoid this, but that’s not evaluated.
  - RHT adds compute/memory traffic. While the paper says it can be fused and is relatively inexpensive for moderate tile sizes (Appendix C), the real cost depends on kernels and overlap.
  - 2D scaling for weights reduces forward/backward mismatch but uses coarser granularity than `1×16`. Thankfully, weights can adapt; still, this is a design tension (Appendix E.5).
- Robustness and generality
  - Most ablations at 12B start mid-training (3.43T tokens) for efficiency (Figure 4), which is reasonable but means some early-training pathologies could be underexplored.
  - The main large-scale proof is on a hybrid Mamba–Transformer. A pure Transformer or gigantic MoE might interact differently with FP4 quantization noise (Section 6 future work).

## 7. Implications and Future Directions
- How this changes the landscape
  - It validates a concrete recipe for “mostly FP4” pretraining at serious scale—10T tokens—while keeping accuracy. That lowers the barrier for frontier training runs and makes FP4 a realistic target for next-gen training efficiency.
  - It also reframes a key failure mode: the “chain-rule violation” caused by forward/backward quantizing the same weight differently. The fix (2D weight scaling) is transferable beyond NVFP4 and should influence other low-precision training designs.
- What follow-ups this enables
  - Push FP4 deeper: reduce the remaining high-precision regions (final blocks, attention GEMMs), test pure Transformers, MoE, and even larger models and longer token horizons (Section 6).
  - Evaluate true system-level wins: measure end-to-end speedups and energy, accounting for RHT, scaling overheads, and improved kernels that fuse transforms/scaling.
  - Better adaptivity: auto-tune which layers remain high precision based on quantization error/gradient stats (Appendix E.2 hints the last layers’ Wgrad errors are larger).
  - Study scaling laws for FP4 formats: how token and parameter scaling curves shift for `NVFP4` vs `MXFP4` vs `FP8` (Section 5 suggests `NVFP4` is more data-efficient).
- Practical applications and downstream use
  - Pretraining at lower cost enables more frequent refreshes of base models, bigger-scale runs on the same budget, or wider access to strong open models.
  - The method should generalize to post-training finetuning and instruction tuning (Section 6 mentions post-training scenarios), potentially making large-batch finetunes cheaper if the same stability tricks hold.

Quotes and figure/table anchors for grounding
- NVFP4’s goal and result:
  - “We validate… training a 12‑billion‑parameter model on 10 trillion tokens… results show… training loss and downstream task accuracies comparable to an FP8 baseline… MMLU‑pro 62.58% vs 62.62%” (Abstract; Section 3; Table 2).
- Loss tracking:
  - “Relative loss error… stays below 1%… widens to slightly above 1.5% during LR decay” (Figure 2 caption and paragraph after).
- Training recipe summary:
  - Four key techniques and ablations showing each matters (Section 4; Figure 4).
- Format comparison:
  - “MXFP4 matches NVFP4 loss when trained on 36% more tokens” (Section 5; Figure 6b).
- NVFP4 format details:
  - Smaller 16‑element blocks, `E4M3` scales, and tensor-level `FP32` scale (Section 2; Figure 1; Appendix B).
- Equations describing scaling/quantization mechanics:
  - Global encode scale (eq. (1)), local block decode scale (eq. (2)), quantized block scale (eq. (3)), element quantization (eq. (4)), GEMM de-scaling (eq. (5)) (Appendix B).
- RHT mechanics:
  - Orthogonality cancellation in GEMMs (eq. (6)); constrained to `Wgrad` in practice (Section 4.2; Appendix C; Appendix E.4.1).

If you just want the TL;DR “how to reproduce” checklist:
- Use `NVFP4` with `16`-element blocks and `E4M3` block scales plus a per-tensor `FP32` scale; quantize GEMM inputs; accumulate in `FP32`.
- Keep ~15% of linear layers in high precision (mostly the last blocks; the first couple can help).
- Use `2D` block scaling (`16×16`) for weights; `1×16` for activations/gradients.
- Apply `Random Hadamard Transforms` on `Wgrad` inputs only (tile size `16×16`; single fixed random sign vector).
- Use `stochastic rounding` for gradients; keep round-to-nearest-even elsewhere.
- If you need to close the final loss gap, switch the forward pass to `BF16` near the start of LR decay.
