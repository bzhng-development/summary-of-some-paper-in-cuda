# LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

**ArXiv:** [2402.13753](https://arxiv.org/abs/2402.13753)

## 🎯 Pitch

LongRoPE presents a breakthrough method to expand the usable context window of large language models from a mere few thousand to over two million tokens—more than 500× previous limits—without requiring massive retraining at ultra-long lengths. By intelligently learning dimension- and position-specific rescalings of rotary position embeddings, coupled with a progressive fine-tuning and interpolation strategy, LongRoPE preserves short-context performance while unlocking unprecedented long-range reasoning. This leap enables LLMs to handle tasks like book-length analysis, multi-document reasoning, and long-form conversations, advancing the capabilities of existing models for real-world, memory-intensive applications.

---

## 1. Executive Summary
LongRoPE is a method for expanding the usable context window of existing large language models (LLMs) from a few thousand tokens to 2,048,000 tokens (about two million) with minimal retraining. It does this by learning how to rescale the model’s rotary position embedding (RoPE) in a non-uniform way across dimensions and token positions, then combining a small amount of fine-tuning (only up to 256k context) with a second interpolation step to reach 2048k.

## 2. Context and Motivation
- Problem addressed
  - Most popular open LLMs (e.g., `LLaMA2`) are trained for short contexts (e.g., 4k tokens). Beyond that, quality drops because the model has not seen those positions during pretraining and the positional encodings are out of distribution (Introduction).
  - Moving from 4k to million-token contexts introduces three hard obstacles (Introduction):
    - Unseen positions create “catastrophic values” that destabilize training.
    - Lack of very long training texts and prohibitive cost to fine-tune at million-token lengths.
    - When context becomes extremely long, attention gets diluted, hurting performance on short contexts.

- Why it matters
  - Many practical tasks require long memory: multi-document reasoning, book-length analysis, codebase understanding, long chats/agent traces, and long-horizon planning. Extending context while preserving short-context ability unlocks these use cases.

- Prior approaches and gaps
  - All these methods work by modifying how positional information (RoPE) is scaled:
    - `PI` (linear positional interpolation) uniformly compresses positions by the extension factor; it degrades at high extensions because positions become too “crowded” (Sec. 2.1; Fig. 2 middle).
    - `NTK`-aware scaling distributes compression unequally across RoPE dimensions; it helps a bit but typically tops out at ≈4× without fine-tuning (Sec. 2.1).
    - `YaRN` mixes interpolation/extrapolation across dimension groups based on hand-crafted rules; it improves some settings but relies on human heuristics and can fail to reach long targets without fine-tuning (Sec. 2.1; Table 1 shows poor PG19 perplexity without fine-tuning).
  - Overall, prior work either requires expensive long-context fine-tuning or breaks down beyond ≈128k.

- Positioning
  - LongRoPE identifies two overlooked “non-uniformities” in positional interpolation and formalizes a search problem to find per-dimension scaling plus a rule for the first tokens that should not be interpolated (Sec. 2.2, Sec. 3). This yields good zero-shot (no fine-tuning) extension up to 8× and serves as a strong initialization for lightweight fine-tuning to 256k, followed by a second interpolation to 2048k (Sec. 3.3).

## 3. Technical Approach
At a high level, LongRoPE learns how to stretch RoPE so that:
- different RoPE dimensions are compressed/extrapolated by different amounts, and
- the first few tokens in a sequence remain uncompressed,
then uses a progressive recipe (search → short fine-tune → second search) to reach million-token windows.

Key building blocks:

1) Background: RoPE and positional interpolation
- `RoPE (Rotary Position Embedding)` encodes token position by rotating query/key vectors dimension-wise with sin/cos at different frequencies. For a token index `n` and RoPE dimension `i`, the “rotation angle” is `n * θ_i` where `θ_i = θ^(−2i/d)` and `θ = 10000` (Eq. 1; Sec. 2.1).
- Extending context by an `extension ratio s = L′ / L` means reusing the original RoPE range up to `L` for positions up to `L′`, so new positions must be “mapped back” by rescaling the rotation angles (Sec. 2.1; Eq. 2).
  - `PI` uses the same scale for all dimensions: divide all angles by `s`.
  - `NTK` scales dimensions unequally, compressing low-frequency dimensions more and high-frequency less.
  - `YaRN` groups dimensions into three frequency bands and applies different rules (some extrapolation, some NTK, some linear). But those groupings are heuristic.

2) The two non-uniformities LongRoPE exploits (Sec. 2.2)
- Non-uniform across RoPE dimensions
  - Not all dimensions carry position information equally. Searching per-dimension scale factors `λ_i` (instead of one global factor) reduces perplexity substantially without any fine-tuning (Table 1: e.g., for LLaMA2-7B at 16k, PG19 perplexity drops from 20.49 with PI to 11.34 with per-dimension search).
- Non-uniform across token positions
  - The earliest tokens in a context tend to receive strong attention throughout the sequence; compressing them hurts. Leaving the first `n̂` tokens uncompressed improves perplexity, and the optimal `n̂` depends on target length (Table 2 shows consistent gains for both PI and NTK at 8k/16k).

3) Problem formulation (Sec. 3.1; Eq. 3)
- Goal: Given a target length `L′`, find per-dimension rescale factors `λ̂_i` and a threshold `n̂` so that:
  - For tokens before `n̂`, you apply the original RoPE angles (no interpolation).
  - For tokens at/after `n̂`, you divide the angle by `λ_i` (equivalently scale frequencies by `1/λ_i`).
- Objective: minimize next-token prediction loss (perplexity) on documents with length ≥ `L′`.
- This turns extension into an optimization over `I(λ̂_i, n̂)` in Eq. 3.

4) Evolutionary search to solve the optimization (Sec. 3.2; Algorithm 1; Table 4)
- Search space
  - `λ_i` ranges from `1.0` (no compression, i.e., pure extrapolation) up to `1.25 × s` (even stronger compression than PI), step `0.01` (Table 4).
  - `n̂ ∈ {0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256}`.
- Practical efficiency tricks
  - Seed the initial population with the actual PI, NTK, and YaRN settings and mutate them, instead of starting random (Sec. 3.2: “Optimized initial population generation”).
  - Enforce a monotonic constraint `λ_i ≤ λ_{i+1}` across dimensions (lower-index/higher-frequency dims get less compression). This reduces poor candidates and speeds up convergence (Sec. 3.2: “Monotonically non-decreasing constraint”), consistent with NTK theory.
- Fitness and selection
  - Evaluate each candidate by applying its RoPE scaling to the target LLM and computing perplexity on long documents (PG19 or Books3 depending on `L′`). Keep the top-k and generate the next population by mutation and crossover (Algorithm 1).

5) 8× extension without fine-tuning (Sec. 3.2; Fig. 3)
- The search often finds solutions that push from 4k to 32k with stable perplexity, while PI/NTK/YaRN degrade quickly after 2× (Fig. 3).

6) Progressive path to 2,048k (Sec. 3.3)
- Step A: Search at 128k and 256k on the pretrained model to get good initial RoPE rescalings.
- Step B: Fine-tune only up to 256k:
  - For `LLaMA2-7B`, fine-tune 400 steps at 128k, then switch the rescale to the searched 256k setting and train 600 more steps (Sec. 4.1; Appendix A.2/Fig. 5 shows loss curves and argues this staged path converges faster than directly jumping to 256k).
  - For `Mistral-7B`, follow YaRN-like settings with a 16k training length and constant LR for 400 steps (Sec. 4.1).
- Step C: Second interpolation (search) on the 256k-fine-tuned model to reach 2048k without further training (a final 8× jump from 256k to 2048k).

7) Short-context recovery (Sec. 3.3; Table 10)
- Extreme compression (512× from 4k to 2048k) “crowds” the short positions and can hurt 4k/8k performance. A final targeted search at short lengths (e.g., 4k, 8k) reduces the maximum allowed `λ` and re-optimizes the RoPE factors for short contexts, applied dynamically at inference if the sequence is short. This substantially improves short-context perplexity and leaderboard accuracy (Table 10).

Implementation note: LongRoPE changes only the positional embedding scaling, so model architecture and most optimizations remain intact (Abstract).

## 4. Key Insights and Innovations
- Dimension-wise and position-wise non-uniform interpolation (fundamental)
  - What’s new: Move from uniform or hand-grouped scaling to fully per-dimension `λ_i` plus a learned “no-interpolation” prefix length `n̂`.
  - Why it matters: It preserves high-frequency RoPE dimensions and critical early tokens, which reduces information loss from compression. Evidence:
    - Without fine-tuning, per-dimension search beats PI, NTK, and YaRN on 8k/16k (Table 1).
    - Keeping the first `n̂` tokens uncompressed consistently improves perplexity; the best `n̂` depends on the target length (Table 2).
- Evolutionary search with monotonic constraint (practical + enabling)
  - What’s new: A scalable search with two key accelerators: seeded population and monotonic `λ_i ≤ λ_{i+1}`.
  - Why it matters: The naive search space is astronomically large; these constraints make it tractable to find high-quality settings, enabling 8× extension without fine-tuning (Algorithm 1; Fig. 3), and quick convergence on long targets (Appendix A.3/Fig. 6).
- Progressive two-stage extension to 2048k (practical + enabling)
  - What’s new: Fine-tune only up to 256k (1k steps total), then apply a second interpolation to reach 2048k without extra training (Sec. 3.3).
  - Why it matters: This avoids the lack of million-token training texts and vastly reduces compute. Table 9 shows the second interpolation is crucial: at 2048k, LongRoPE has perplexity 7.08 vs 8.27 (YaRN) and 20.17 (PI).
- Short-context performance recovery (practical)
  - What’s new: A final targeted search for 4k/8k on the already-extended model; switch RoPE factors at inference time if the sequence is short (Sec. 3.3).
  - Why it matters: It mitigates the common short-context degradation seen with aggressive interpolation (Table 10 shows 4k perplexity drops from 4.51→3.85 and average leaderboard accuracy improves from 47.9→50.8 for the 256k-fine-tuned LLaMA2-2048k).

## 5. Experimental Analysis
- Evaluation setup (Sec. 4.1)
  - Models: `LLaMA2-7B` and `Mistral-7B`.
  - Tasks:
    - Long-sequence language modeling: perplexity on PG19 and Proof-Pile/Books3. Perplexity is a standard metric where lower is better.
    - Passkey retrieval: find a 5-digit number hidden in long text; measures whether the model can use the entire context (Appendix A.1 includes the prompt template).
    - Short-context benchmarks: ARC-C (25-shot), HellaSwag (10-shot), MMLU (5-shot), TruthfulQA (0-shot) (Table 8).
  - Baselines: Together-32k (PI), LongLoRA-100k (PI), CodeLLaMA-100k (NTK), YaRN models at 64k and 128k for both LLaMA2 and Mistral (Tables 5–7).

- Main quantitative results
  - Within 256k (Proof-pile; Table 5)
    - LLaMA2-7B, perplexity at 262,144 tokens:
      > LongRoPE-2048k (ft=256k): 1.87 vs CodeLLaMA-100k (NTK) 49.33 and YaRN-128k 99.64.
    - Mistral-7B, perplexity at 262,144 tokens:
      > LongRoPE-2048k: 1.84–1.85 vs YaRN-128k 4.91.
    - Trend: LongRoPE’s perplexity generally decreases as evaluation context increases from 4k→256k, indicating it actually uses longer context (Sec. 4.2).
  - Beyond 2000k (Books3; Table 6)
    - LLaMA2-7B:
      > LongRoPE-2048k (ft=256k): 6.17 (256k), 6.17 (512k), 6.35 (1024k), 7.08 (2048k), while baselines blow up past 128k (e.g., PI 246.45 at 256k).
    - Mistral-7B:
      > LongRoPE-2048k: 6.68–7.15 at 256k; then increases to 12.78–13.71 at 2048k. So it extends but with higher perplexity than LLaMA2 at extreme lengths.
  - Passkey retrieval (Fig. 4)
    - LLaMA2-2048k (ft=256k) maintains ≥90% accuracy from 4k up to 2048k, while all baselines drop to near 0 beyond 128k.
    - Mistral-2048k (ft=128k) holds 100% up to ~1800k, then 60% at 2048k.
  - Standard benchmarks at short length (Table 8)
    - LLaMA2-7B:
      > Original vs LongRoPE-2048k (ft=128k): HellaSwag 78.6 → 76.5, MMLU 46.6 → 43.4; relatively close. The ft=256k variant drops more (e.g., MMLU 39.6).
    - Mistral-7B:
      > TruthfulQA slightly improves: 42.6 (orig) → 43.1 (LongRoPE-2048k ft=128k). Other tasks are close, minor degradation (~1–2 points).
  - Ablations and diagnostics
    - Second interpolation matters (Table 9): On LLaMA2-256k, at 2048k LongRoPE 7.08 vs YaRN 8.27 and PI 20.17.
    - Short-context recovery works (Table 10): For LLaMA2-2048k (ft=256k), 4k perplexity improves from 4.51 → 3.85 and leaderboard average from 47.9 → 50.8.
    - Contribution of the two non-uniformities (Table 11): Per-dimension scaling drives most gains; adding the `n̂`-token rule further helps at 16k/32k, but is less impactful at 2048k.
    - Search efficiency (Appendix A.3, Fig. 6): Validation perplexity drops quickly in early iterations; YaRN can underperform PI at large extension (64×), highlighting the benefit of search over heuristics.

- Do the results support the claims?
  - Yes, in three ways:
    - Stable perplexity and passkey retrieval up to 2048k (Tables 5–7; Fig. 4).
    - Only 1k fine-tuning steps at ≤256k are needed before the second interpolation to reach 2048k (Sec. 3.3; Sec. 4.1; Appendix A.2).
    - Competitive short-context performance after targeted recovery (Table 10; Table 8).
  - Mixed areas:
    - Mistral’s performance degrades more beyond 256k (Table 6, Fig. 4), which the authors attribute to its fine-tuning regime using 16k training sequences (Sec. 4.2 discussion).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Works for models using RoPE; does not address absolute or learned positional embeddings without RoPE (Sec. 1).
  - The method optimizes for perplexity on long documents, which correlates with but does not guarantee gains on all downstream tasks.
- Compute and data
  - Although far cheaper than full million-token fine-tuning, the approach still uses substantial compute:
    - LLaMA2-128k fine-tuning: 8×A100 for ~1 week (400 steps); LLaMA2-256k: 16×A100 for ~2 weeks (600 steps) (Appendix A.2).
    - Search costs grow with target length: evaluation at 2048k takes ~50 minutes per perplexity call; searches are capped to ~5 days on 8×A100 (Appendix A.3).
- Model-specific tuning
  - The searched `λ_i` and `n̂` are per-model and per-target-length; porting to another model/length requires re-running the search.
- Short-context trade-off
  - Extreme extension “crowds” the short positions (Sec. 3.3). The paper mitigates this with short-context readjustment, but residual degradation can remain (Table 8 shows modest drops on MMLU/HellaSwag for LLaMA2).
- Generalization to other modalities/tasks
  - The evaluation focuses on text modeling/perplexity and a synthetic retrieval task. It does not study tasks like complex multi-turn tool use over millions of tokens or cross-modal settings.

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that existing 7B LLMs can be made to reason over million-token contexts with minimal architectural changes and relatively modest fine-tuning. This shifts the boundary of what “in-context learning” can handle—e.g., full books, large code repositories, legal corpora, or long agent memories.
- Enabled directions
  - Better searches: Content-aware or layer-wise RoPE scaling; Bayesian/gradient-based search to reduce evaluation cost; multi-objective search balancing short- and long-context performance.
  - Theory: Why do specific RoPE dimensions matter more? Can we characterize the information distribution across dimensions and positions (Findings in Sec. 2.2)?
  - Training recipes: Curriculum over lengths or synthetic datasets that mimic long-range structure to reduce fine-tuning even further; combining with efficient training (e.g., LongLoRA/PoSE).
  - Complementary methods: Integrate with retrieval/memory systems (Related Work Sec. 5) and streaming attention to combine million-token context with external knowledge.
- Practical applications
  - Long-document QA and summarization; codebase refactoring and analysis; long-horizon planning in agents; scientific literature synthesis; legal/contract analysis with minimal chunking; lifelong conversation histories.
  
Selected evidence highlights
- “8× extension without fine-tuning” with better perplexity than PI/NTK/YaRN (Fig. 3; Table 1).
- Progressive recipe: 400 steps at 128k + 600 steps at 256k, then a second interpolation to 2048k (Sec. 3.3; Sec. 4.1; Appendix A.2).
- LLaMA2 Books3 perplexity: 6.17 (256k) → 6.17 (512k) → 6.35 (1024k) → 7.08 (2048k) (Table 6).
- Passkey retrieval ≥90% through 2048k for LLaMA2 (Fig. 4).
- Short-context recovery improves both perplexity and benchmark accuracy (Table 10).

In short, LongRoPE’s core insight—treating positional scaling as a learnable, non-uniform, per-dimension-and-position transformation—combined with an efficient search and a progressive extension schedule, makes million-token context windows feasible on today’s LLMs while maintaining practical performance at short lengths.
