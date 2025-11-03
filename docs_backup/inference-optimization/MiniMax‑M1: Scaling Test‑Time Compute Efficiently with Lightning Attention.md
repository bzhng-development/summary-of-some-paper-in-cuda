# MiniMax‑M1: Scaling Test‑Time Compute Efficiently with Lightning Attention

**ArXiv:** [2506.13585](https://arxiv.org/abs/2506.13585)
**Authors:** MiniMax, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, Chengjun Xiao, Chengyu Du, Chi Zhang, Chu Qiao, Chunhao Zhang, Chunhui Du, Congchao Guo, Da Chen, Deming Ding, Dianjun Sun, Dong Li, Enwei Jiao, Haigang Zhou, Haimo Zhang, Han Ding, Haohai Sun, Haoyu Feng, Huaiguang Cai, Haichao Zhu, Jian Sun, Jiaqi Zhuang, Jiaren Cai, Jiayuan Song, Jin Zhu, Jingyang Li, Jinhao Tian, Jinli Liu, Junhao Xu, Junjie Yan, Junteng Liu, Junxian He, Kaiyi Feng, Ke Yang, Kecheng Xiao, Le Han, Leyang Wang, Lianfei Yu, Liheng Feng, Lin Li, Lin Zheng, Linge Du, Lingyu Yang, Lunbin Zeng, Minghui Yu, Mingliang Tao, Mingyuan Chi, Mozhi Zhang, Mujie Lin, Nan Hu, Nongyu Di, Peng Gao, Pengfei Li, Pengyu Zhao, Qibing Ren, Qidi Xu, Qile Li, Qin Wang, Rong Tian, Ruitao Leng, Shaoxiang Chen, Shaoyu Chen, Shengmin Shi, Shitong Weng, Shuchang Guan, Shuqi Yu, Sichen Li, Songquan Zhu, Tengfei Li, Tianchi Cai, Tianrun Liang, Weiyu Cheng, Weize Kong, Wenkai Li, Xiancai Chen, Xiangjun Song, Xiao Luo, Xiao Su, Xiaobo Li, Xiaodong Han, Xinzhu Hou, Xuan Lu, Xun Zou, Xuyang Shen, Yan Gong, Yan Ma, Yang Wang, Yiqi Shi, Yiran Zhong, Yonghong Duan, Yongxiang Fu, Yongyi Hu, Yu Gao, Yuanxiang Fan, Yufeng Yang, Yuhao Li, Yulin Hu, Yunan Huang, Yunji Li, Yunzhi Xu, Yuxin Mao, Yuxuan Shi, Yuze Wenren, Zehan Li, Zelin Li, Zhanxu Tian, Zhengmao Zhu, Zhenhua Fan, Zhenzhen Wu, Zhichao Xu, Zhihang Yu, Zhiheng Lyu, Zhuo Jiang, Zibo Gao, Zijia Wu, Zijian Song, Zijun Sun
**Institutions:** MiniMax AI (likely MiniMax)

## 🎯 Pitch

MiniMax-M1 revolutionizes large scale reasoning by integrating a hybrid Lightning/softmax attention system and the novel `CISPO` reinforcement learning algorithm. This development allows for efficient processing of up to 1M-token contexts and 80K-token outputs, enabling applications in complex software engineering and long-document comprehension, all while maintaining competitive open-weight performance.

---

## 1. Executive Summary
MiniMax-M1 introduces a large reasoning model that makes very long “thinking at inference time” practical by combining a hybrid attention stack (mostly linear-time “Lightning Attention” with periodic softmax attention) and a new reinforcement learning algorithm, `CISPO`, that stabilizes learning for long chains of thought. The result is an open-weight system with 1M-token input context and up to 80K-token generated reasoning, achieving strong efficiency—near-linear compute growth with sequence length—and competitive or leading performance on software engineering, tool use, and long-context benchmarks (Figure 1, Table 2).

## 2. Context and Motivation
- Problem addressed
  - Reasoning-focused LLMs benefit from “test-time compute” scaling—letting the model think for more tokens improves quality. But standard softmax attention grows quadratically with sequence length, making very long outputs and contexts prohibitively expensive. This hinders both inference and reinforcement learning (RL) training that encourages long reasoning (Introduction; Figure 1 Right).
- Why it matters
  - Real applications (complex software engineering, multi-step tool use, long documents) demand long inputs and long, reflective reasoning traces. Efficiently scaling compute at test time enables these capabilities at reasonable cost (Introduction, §4 Software Engineering; Long Context section).
- Prior approaches and limitations
  - Many efficient attention or sequence models exist—sparse attention, linear attention variants, state space models, and linear RNNs (Introduction). Yet, competitive large reasoning models still predominantly use softmax attention; few have validated efficient architectures at large scale with open weights. The notable non-open exception is Hunyuan-T1 (Mamba-based).
- Positioning
  - MiniMax-M1 is built on MiniMax-Text-01 and adopts:
    - A hybrid attention stack: predominantly Lightning Attention (a linear-attention variant with I/O-aware implementation) interleaved with periodic softmax attention to preserve global interactions (§1, §3.2).
    - Large-scale RL with a new algorithm (`CISPO`) and a diverse set of verifiable and model-judged tasks (§3–§5).
  - It targets long-context and long-generation efficiency and robustness while remaining open weight (GitHub/HuggingFace links in §1).

Key definitions used throughout:
- `Test-time compute`: the FLOPs the model spends during inference, especially for long chains of thought.
- `Context length`: the maximum number of input tokens the model can condition on.
- `Thinking budget`: the maximum number of output tokens the model is allowed to generate during “reasoning.”
- `MoE (Mixture-of-Experts)`: an architecture where only a subset of “experts” (sub-networks) are activated per token, reducing the active parameters per token compared to the total parameter count.

## 3. Technical Approach
This section unpacks the architecture and training pipeline and explains the `CISPO` algorithm.

- Model architecture
  - Base and size: built on MiniMax-Text-01, with 456B total parameters and 32 experts, but only 45.9B parameters “activated” (used) per token (§1).
  - Hybrid attention design:
    - Lightning Attention blocks (a linear-attention family with I/O-aware kernels; see Qin et al., 2022a; 2024b,c) provide near-linear scaling with sequence length.
    - Every 7 Lightning/“transnormer” blocks are followed by 1 standard softmax-attention transformer block (§1).
    - Rationale: the linear blocks give efficiency, while occasional softmax blocks help maintain precise global interactions and mitigate approximation drift (§1). Figure 1 (Right) plots theoretical FLOPs vs. generation length, showing much slower growth than full-softmax competitors: “less than 50% FLOPs at 64K tokens and ~25% at 100K tokens vs. DeepSeek R1.”
  - Context and output lengths:
    - Native input context up to 1M tokens; released models allow 40K and 80K output budgets (Table 1; §5).

- Data pipeline and pretraining/fine-tuning
  - Continual pretraining on 7.5T tokens emphasizing reasoning-heavy content (≈70% STEM, code, books, reasoning) (§2.1). Training used an initial constant LR of 8e-5 for 2.5T tokens, then decayed to 8e-6 over 5T tokens (Training Recipe in §2.1).
  - Long-context extension strategy: incrementally extended context over four stages from 32K to 1M to avoid optimization instability, given different decay rates across Lightning layers that make early layers more local (§2.1 Long Context Extension).
  - Supervised fine-tuning (SFT): injects reflection-style chain-of-thought (CoT) on curated long-CoT data across math, coding, STEM, writing, QA, and chat (≈60% math/coding) (§2.2). Purpose: provide a strong starting policy for RL.

- Reinforcement learning with `CISPO`
  - Motivation: Standard PPO/GRPO “clip” token-level updates when the policy’s probability ratio `r` becomes large. In practice, rare but crucial “reflection” tokens (e.g., “However,” “Recheck,” “Wait,” “Aha”) often have very low base probabilities, leading to large ratios and thus being clipped away—preventing these tokens from contributing to off-policy gradient steps (§3.1 Issues of Token Clipping).
  - Background notation
    - For a prompt `q`, the policy `πθ` produces a response `o`. PPO uses token-level ratios `ri,t = πθ(oi,t | q, oi,<t) / πθold(oi,t | q, oi,<t)` and clips the update (Eq. 1).
    - GRPO replaces a learned value function with a group-relative advantage: `Âi,t = (Ri − mean({Rj})) / std({Rj})` where G responses are sampled per question (Eq. 2).
  - CISPO objective (Eq. 4–5)
    - Start from REINFORCE with importance sampling (IS) and stop-gradient on IS weights (Eq. 3).
    - Replace PPO/GRPO’s token clipping with clipping of the IS weights directly:
      - ˆri,t(θ) = clip(ri,t(θ), 1 − εIS_low, 1 + εIS_high) (Eq. 5)
      - In experiments they set no lower bound (εIS_low large) and tune only the upper bound (§3.1).
    - Loss averages token-level `log πθ` weighted by stop-gradient clipped IS weights times group-relative advantages (Eq. 4):
      - Intuition: never “drop” any token’s gradient entirely; instead, cap how much any one token can magnify the update. This preserves learning on rare reflection tokens while stabilizing variance.
    - Generalized view (Eq. 6–7): Introduces a mask Mi,t that recovers PPO’s trust-region behavior as a special case—providing a unifying formulation where clipping can be applied either to weights (CISPO) or to token updates (PPO-style).
    - Practical details: no explicit KL penalty; uses dynamic sampling and length penalties (as in Yu et al., 2025) (§3.1).

- Engineering for stable, efficient RL on the hybrid architecture (§3.2)
  - Precision mismatch fix: training-mode and inference-mode probabilities diverged due to precision in the LM head. Upgrading the LM head to FP32 realigned them (Figure 3). Pearson correlation improved from ~0.987 to ~0.997 and remained stable through training, enabling rewards to increase.
  - Optimizer settings: Gradients mostly < 1e−14 and weakly correlated between steps. AdamW with β1=0.9, β2=0.95, ε=1e−15 worked better than common defaults (§3.2).
  - Early truncation of pathological loops: If 3,000 consecutive tokens each have probability >0.99, halt generation. This avoids runaway repetition that destabilizes optimization and wastes compute (§3.2).

- RL tasks and rewards (§4)
  - Rule-verified environments (§4.1): reward is correctness (plus a format reward).
    - Math: ~50K curated, competition-level problems, filtered for difficulty using pass@10; overlap with SFT removed; benchmark leakage filtered (§4.1).
    - Logical reasoning: ~53K synthetic tasks spanning 41 types (e.g., ciphers, Sudoku) via the SynLogic generator with task-specific verifiers and adaptive difficulty bounds (pass@10 > 0 and 0–0.5 on base model) (§4.1).
    - Competitive programming: ~30K problems from judge sites; where absent, test suites synthesized; filtered by model pass rates to keep moderate difficulty (§4.1).
    - Software engineering: sandboxed real GitHub issues/PRs with execution-based rewards (pass/fail on tests) for tasks like bug localization/fixing and test synthesis; several thousand high-quality samples (§4.1).
  - Model-judged general tasks (§4.2):
    - With ground truth but hard to parse: use a Generative Reward Model (`GenRM`) with a five-grade scale; evaluate via human-annotated benchmarks and Best-of-N selection behavior (§4.2.1).
    - Without ground truth: pairwise preference rewards (−1/0/1) vs. vetted reference answers; for constrained instruction-following, combine rule-based and model-based rewards (§4.2.1).
    - Mitigating length bias in reward models: continuous online monitoring for reward hacking via overlong outputs; when detected, recalibrate `GenRM` and apply RL-side reward shaping, clipping, and normalization (§4.2.2).
  - Curriculum (§4.3): begin RL with verifiable reasoning tasks; gradually mix general-domain tasks. Aim: retain rigorous skills (math/code) while learning flexible general reasoning.

- Extending thinking from 40K → 80K tokens (§5)
  - Data filtering: use the 40K model to identify “too easy” examples and shift the distribution toward harder math/coding; downsample synthetic reasoning that destabilized long-context RL (§5 Data).
  - Staged window expansion: 40K → 48K → 56K → 64K → 72K → 80K, advancing when perplexity stabilizes and the 99th-percentile output length approaches the limit (§5 Length Scaling Strategy).
  - Preventing “pattern collapse” late in windows: identify root cause (negatives reaching the length limit faster → large negative gradients concentrated later); apply three remedies—(1) early stopping via repetition detection; (2) blended sample-level loss with token-level normalization; (3) lower gradient clipping threshold and a smaller εIS_high (§5).

Compute footprint: Full RL training completed in 3 weeks on 512 H800 GPUs; estimated rental cost ≈ $534,700 (§1).

## 4. Key Insights and Innovations
- Hybrid Lightning/softmax attention that scales test-time compute efficiently
  - What’s new: A deep stack of Lightning Attention blocks with periodic softmax blocks (1 in 8) to preserve global fidelity (§1). Prior works rarely validate linear-attention variants at this scale for reasoning models.
  - Why it matters: Figure 1 (Right) shows theoretical FLOPs grow much more slowly with generation length than full-softmax models—“<50% of DeepSeek R1’s FLOPs at 64K tokens; ~25% at 100K tokens.” This enables 1M-token inputs and 80K-token outputs without quadratic blowup (Table 1).
- `CISPO`: clip importance weights instead of token updates
  - What’s new: Replace PPO/GRPO’s clipping of token updates with clipping of IS weights (Eq. 4–5). This preserves gradients from rare, high-impact tokens (reflection “forks”) that PPO-style clipping discards (§3.1).
  - Why it matters: On Qwen2.5-32B with a math RL setup, CISPO reaches DAPO’s AIME 2024 accuracy using half the training steps and outperforms GRPO at matched steps (Figure 2), indicating better sample efficiency for long CoT.
- RL stability recipes for hybrid linear attention at scale
  - What’s new: Identify and fix a training/inference precision mismatch by moving LM head to FP32 (Figure 3), tune AdamW for tiny gradients, and add early truncation of high-probability repetition (§3.2).
  - Why it matters: Without these, rewards stagnated or training diverged. The fixes make long-output RL feasible and stable for the hybrid architecture.
- Verifiable SE sandbox RL and long-context competence
  - What’s new: Execution-based rewards on real-world repositories (compile, run tests) extend beyond typical math/programming RL (§4.1). Long-context support reaches 1M tokens.
  - Why it matters: The model attains strong SWE-bench Verified results and leading long-context retrieval at 128K and competitive performance at 1M (Table 2), suggesting practical readiness for large codebases and lengthy documents.

## 5. Experimental Analysis
- Evaluation setup (§6)
  - Sampling: temperature 1.0, top-p 0.95 for all tasks.
  - Benchmarks and metrics:
    - Math: AIME 2024/2025 (pass@32 averaged), MATH-500 (§6.1).
    - Coding: LiveCodeBench (24/8–25/5 window) and FullStackBench, report pass rates over 16 samples (§6.1).
    - Reasoning & Knowledge: GPQA-Diamond (pass@32), MMLU-Pro, ZebraLogic, HLE text-only (§6.1).
    - Software engineering: SWE-bench Verified, using the Agentless scaffold with a two-stage non-embedding localization pipeline (§6.1).
    - Long context: OpenAI-MRCR at 128K and 1M words, LongBench-v2 (§6.1).
    - Tool use: TAU-bench airline and retail domains (GPT-4.1 as user model, no custom tools, 40-step limit; footnote 2 in §6.1).
    - Factuality: SimpleQA.
    - General assistant: MultiChallenge (GPT-4o judged).
- Main quantitative results (Table 2; Figure 1 Left)
  - Long-context understanding
    - MRCR (128K): `MiniMax-M1-80k` 73.4, above o3 56.5, Claude 4 Opus 48.9, DeepSeek-R1-0528 51.5; close to Gemini 2.5 Pro 76.8.
    - MRCR (1M): `MiniMax-M1-80k` 58.6; Gemini 2.5 Pro 58.8. This confirms effective 1M-token context handling.
    - LongBench-v2: `MiniMax-M1-80k` 61.5 vs DeepSeek-R1-0528 52.1 and Qwen3-235B 50.1.
  - Software engineering
    - SWE-bench Verified: `MiniMax-M1-80k` 56.0; behind DeepSeek-R1-0528 57.6 but well above other open weights (Table 2).
  - Tool use
    - TAU-bench airline: `MiniMax-M1-80k` 62.0; better than DeepSeek-R1-0528 53.5 and Qwen3-235B 34.7; near Claude 4 Opus 59.6 (Table 2).
    - TAU-bench retail: `MiniMax-M1-80k` 63.5; trails Claude 4 Opus 81.4 and DeepSeek-R1-0528 63.9; near MiniMax-M1-40k 67.8 (Table 2).
  - Math/coding
    - AIME 2024: `MiniMax-M1-80k` 86.0; behind DeepSeek-R1-0528 91.4; above the original DeepSeek-R1 79.8 (Table 2).
    - AIME 2025: `MiniMax-M1-80k` 76.9 vs DeepSeek-R1-0528 87.5.
    - LiveCodeBench: `MiniMax-M1-80k` 65.0 vs DeepSeek-R1-0528 73.1; better than Qwen3-235B 65.9 is comparable; FullStackBench 68.3 comparable to leading models.
  - Knowledge/reasoning
    - GPQA-Diamond: `MiniMax-M1-80k` 70.0; trails DeepSeek-R1-0528 81.0 (Table 2).
    - ZebraLogic: `MiniMax-M1-80k` 86.8; above Qwen3-235B 80.3; close to Claude 4 Opus 95.1.
    - MMLU-Pro: `MiniMax-M1-80k` 81.1 vs DeepSeek-R1-0528 85.0.
  - Factuality and general assistant
    - SimpleQA: `MiniMax-M1-80k` 18.5 exceeds most open weights but trails DeepSeek-R1 27.8 (Table 2).
    - MultiChallenge: `MiniMax-M1-80k` 44.7, near DeepSeek-R1-0528 45.0 and Claude 4 Opus 45.8.

- Evidence for efficiency and RL scaling
  - FLOPs scaling: Figure 1 (Right) shows MiniMax-M1’s theoretical FLOPs rising much more slowly with generation length than DeepSeek-R1 or Qwen3-235B—key for 80K-token reasoning.
  - CISPO effectiveness: On Qwen2.5-32B, CISPO outperforms GRPO and matches DAPO with half the steps on AIME 2024 in a controlled RL setup (Figure 2).
  - Stability fixes: Probability alignment improvement (Figure 3) strongly indicates the FP32 LM head resolves training/inference mismatch.
  - Length vs accuracy: During RL, accuracy rises with generated length on AIME 2024/2025 and LiveCodeBench v5; e.g., AIME 2024 improves from ~68% to ~80% as average outputs exceed 20K tokens (Figure 4).

- Are claims supported?
  - Yes for efficiency and long-context/tool-use/SE strengths: Theory curves (Figure 1 Right) plus strong 128K/1M MRCR and SWE-bench Verified scores support the efficiency and application claims.
  - Mixed for math/coding SOTA: The model generally trails the latest DeepSeek-R1-0528 on AIME/LiveCodeBench, but is competitive with other strong open weights (Table 2).
  - Ablations and robustness:
    - Algorithm ablation: GRPO/DAPO vs CISPO (Figure 2).
    - Engineering ablation: probability correlation before/after precision fix (Figure 3).
    - Scaling study: staged length expansion and remedies against pattern collapse (§5).

> Quote on efficiency: “Compared to DeepSeek R1, M1 consumes less than 50% of the FLOPs at a generation length of 64K tokens, and approximately 25% at a length of 100K tokens.” (Figure 1 Right)

## 6. Limitations and Trade-offs
- Accuracy trade-offs vs frontier models
  - On math/coding competitions, `MiniMax-M1-80k` lags behind DeepSeek-R1-0528 (Table 2). The hybrid linear-attention design, while efficient, may still sacrifice some exactness that specialized softmax-heavy pipelines achieve on these domains.
- Theoretical vs realized efficiency
  - Figure 1 (Right) plots theoretical FLOPs. While indicative, real-world latency/throughput can depend on hardware, kernel maturity, and batching. The paper emphasizes Lightning Attention’s I/O-aware design, but systematic wall-clock speedups vs baselines are not reported in Table 2.
- Training complexity and resource requirements
  - Although “efficient” for the scale, the full RL run used 512 H800 GPUs for 3 weeks (§1). Many organizations cannot reproduce this; however, open weights mitigate some barriers for inference use.
- Reward-model dependence and bias
  - The general RL portion relies on `GenRM`. The work documents length-bias issues and proposes online recalibration and reward shaping (§4.2.2), but reward hacking remains a general risk when training against learned judges.
- Stability sensitivities
  - The training required careful optimizer hyperparameters (β2=0.95, ε=1e−15), FP32 in the LM head, and special truncation rules (§3.2). This suggests the method may be sensitive to implementation details, especially at longer lengths and with hybrid attention.
- Mixed impact of longer thinking
  - Scaling to 80K thinking improved hard math/coding but introduced risks like pattern collapse that required extra mitigation (§5). Some long-context scores (e.g., MRCR 128K) slightly declined from 76.1 (40k) to 73.4 (80k) in Table 2—indicating task-dependent trade-offs.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that large-scale, open-weight reasoning models can use predominantly linear-time attention to support 1M-token contexts and very long outputs while remaining competitive with leading open weights. This directly enables long-document comprehension, large-repository code tasks, and extended multi-tool workflows without quadratic cost growth (Table 1; Figure 1 Right).
- What it enables next
  - Research:
    - Controlled studies of hybrid attention ratios and placements (e.g., 7:1 schedule) for varying domains.
    - Improved reward models robust to length/style bias; calibration during RL rather than post hoc (§4.2.2).
    - Better credit assignment for long CoT—combining CISPO with token- or span-level verifiers to reduce variance.
    - Analysis of the interaction between Lightning Attention’s decay behaviors and RL-induced long-horizon reasoning (§2.1).
  - Systems:
    - Practical agent frameworks leveraging 1M-token memory and 80K-token reflection: enterprise workflows, scientific literature review and planning, long-horizon SE agents with execution feedback (§7 Conclusion; references to real-world agent benchmarks).
    - Cost-aware inference controllers that adjust “thinking budget” on-the-fly using M1’s favorable FLOPs scaling (Figure 1 Right).
- Practical applications
  - Software engineering agents that localize bugs and patch code in real repositories with execution feedback (SWE-bench Verified, §4.1 and Table 2).
  - Long-document retrieval and disambiguation (MRCR 128K/1M, Table 2).
  - Policy- and tool-constrained assistants (TAU-bench) that must reason over long, multi-step interactions without custom tools (Table 2).

In sum, MiniMax-M1 contributes a practically validated path to scale test-time reasoning through a hybrid Lightning/softmax architecture and a learning algorithm, `CISPO`, that preserves gradients from rare but pivotal reasoning tokens. The model excels in long-context, tool-use, and software-engineering tasks, while math/coding competition performance is strong but not state-of-the-art. The approach opens fertile ground for efficient long-horizon agents and suggests rich follow-ups in stable RL for very long sequences.
