# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

**ArXiv:** [2508.06471](https://arxiv.org/abs/2508.06471)
**Authors:** GLM‑4.5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, Yean Cheng, Yifan An, Yilin Niu, Yuanhao Wen, Yushi Bai, Zhengxiao Du, Zihan Wang, Zilin Zhu, Bohan Zhang, Bosi Wen, Bowen Wu, Bowen Xu, Can Huang, Casey Zhao, Changpeng Cai, Chao Yu, Chen Li, Chendi Ge, Chenghua Huang, Chenhui Zhang, Chenxi Xu, Chenzheng Zhu, Chuang Li, Congfeng Yin, Daoyan Lin, Dayong Yang, Dazhi Jiang, Ding Ai, Erle Zhu, Fei Wang, Gengzheng Pan, Guo Wang, Hailong Sun, Haitao Li, Haiyang Li, Haiyi Hu, Hanyu Zhang, Hao Peng, Hao Tai, Haoke Zhang, Haoran Wang, Haoyu Yang, He Liu, He Zhao, Hongwei Liu, Hongxi Yan, Huan Liu, Huilong Chen, Ji Li, Jiajing Zhao, Jiamin Ren, Jian Jiao, Jiani Zhao, Jiaqi Wang, Jiayi Gui, Jiayue Zhao, Jie Liu, Jijie Li, Jing Li, Jing Lu, Jingsen Wang, Jingwei Yuan, Jingxuan Li, Jingzhao Du, Jinhua Du, Jinxin Liu, Junkai Zhi, Junli Gao, Ke Wang, Lekang Yang, Liang Xu, Lin Fan, Lindong Wu, Lintao Ding, Lu Wang, Man Zhang, Minghao Li, Minghuan Xu, Mingming Zhao, Mingshu Zhai, Pengfan Du, Qian Dong, Shangde Lei, Shangqing Tu, Shangtong Yang, Shaoyou Lu, Shijie Li, Shuang Li, Shuang‑Li, Shuxun Yang, Sibo Yi, Tianshu Yu, Wei Tian, Weihan Wang, Wenbo Yu, Weng Lam Tam, Wenjie Liang, Wentao Liu, Xiao Wang, Xiaohan Jia, Xiaotao Gu, Xiaoying Ling, Xin Wang, Xing Fan, Xingru Pan, Xinyuan Zhang, Xinze Zhang, Xiuqing Fu, Xunkai Zhang, Yabo Xu, Yandong Wu, Yida Lu, Yidong Wang, Yilin Zhou, Yiming Pan, Ying Zhang, Yingli Wang, Yingru Li, Yinpei Su, Yipeng Geng, Yitong Zhu, Yongkun Yang, Yuhang Li, Yuhao Wu, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yuxuan Zhang, Zezhen Liu, Zhen Yang, Zhengda Zhou, Zhongpei Qiao, Zhuoer Feng, Zhuorui Liu, Zichen Zhang, Zijun Yao, Zikang Wang, Ziqiang Liu, Ziwei Chai, Zixuan Li, Zuodong Zhao, Wenguang Chen, Jidong Zhai, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, Jie Tang
**Institutions:** Zhipu AI, Tsinghua University

## 🎯 Pitch

GLM-4.5 introduces an innovative Mixture-of-Experts large language model that excels in agentic tool use, deliberate reasoning, and real-world coding through a hybrid "thinking vs. direct response" framework. By open-sourcing a system that rivals proprietary models in these multifaceted tasks, GLM-4.5 paves the way for automated, reliable solutions across diverse domains, offering significant advancements in efficiency and capability for both scientific and practical applications.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model that targets a unified trifecta of capabilities—agentic tool use, complex reasoning, and real-world coding—through a hybrid “thinking vs. direct response” paradigm. A 355B-parameter MoE (32B active per token) and a smaller 106B variant are trained on 23T tokens with a multi-stage pipeline plus extensive reinforcement learning (RL) and expert-model distillation, yielding top-tier results across 12 benchmarks (Figure 1; Tables 3–6), long context up to 128K tokens, and strong parameter efficiency (Figure 2).

## 2. Context and Motivation
- Problem/gap addressed
  - Building one model that is simultaneously strong at:
    - Agentic behaviors (tool use, multi-step interactions with web, APIs, terminals).
    - Deliberate multi-step reasoning (math, science, logic).
    - Software engineering tasks (editing repositories, passing tests, shell/terminal workflows).
  - Open-source models often excel in just one or two of these areas; proprietary models (e.g., OpenAI o1/o3, Claude Sonnet 4) have shown strong task-specific performance but are closed (Introduction; Figure 1).
- Why this matters
  - Practical impact: Agents that browse the web, call tools reliably, and fix real repositories can automate non-trivial work. Strong math/reasoning boosts reliability and planning, while robust coding enables self-correction (Section 3.3; Table 5).
  - Scientific value: Understanding how architecture, data, long-context training, and RL interact to scale reasoning and agentic performance (Sections 2–3).
- Prior approaches and their limits
  - Large MoE models like DeepSeek-V3 and Kimi K2 scale parameters, but open models that are simultaneously strong on agentic + reasoning + coding (“ARC”) are scarce (Table 1; Figure 1).
  - Some RL practices for reasoning use multi-stage length schedules that may degrade long-context ability (Figure 6), and tool-calling pipelines often suffer from brittle formatting (Figure 4).
- Positioning
  - GLM-4.5 provides:
    - A deeper (more layers), narrower (smaller width) MoE design with Grouped-Query Attention (GQA), QK-Norm, and more attention heads to favor reasoning (Section 2.1; Table 1).
    - A multi-stage training pipeline (23T tokens) and post-training that first builds domain experts (reasoning/agent/general), then unifies them via distillation and RL, enabling “thinking” and “non-thinking” modes (Sections 3.1–3.4).
    - System-level RL infrastructure (Slime) that scales asynchronous, long-horizon agent rollouts, with FP8-accelerated inference for fast data generation (Section 3.5; Figure 10).

## 3. Technical Approach
Step-by-step overview, from architecture to training to RL and infrastructure.

- Architecture (Section 2.1; Table 1)
  - MoE with loss-free balance routing: a gating mechanism routes each token to 8 of 160 experts (plus 1 shared), but uses bias updates (not an auxiliary load-balancing loss) during most pre-training to spread load; an auxiliary sequence-level balance loss (weight 1e-4) adds stability later (Section 2.4).
    - Definition: `MoE` activates only a subset of experts per token; “activated parameters” counts only the parameters used for a single token’s forward pass (here 32B for GLM-4.5).
  - Deeper, narrower stack: 89 MoE layers plus 3 dense layers for GLM-4.5; 45 MoE + 1 dense for GLM-4.5-Air. The paper reports that deeper models improved reasoning capacity vs. shallower, wider alternatives (Section 2.1).
  - Attention: Grouped-Query Attention (few key-value heads, many query heads) with partial RoPE; 96 heads at 5120 hidden dim (2.5× typical head count), which did not lower training loss but consistently improved reasoning benchmarks (Section 2.1).
    - `QK-Norm`: normalizes queries/keys to stabilize attention logits.
  - Long context: RoPE base frequency lifted from 10,000 to 1,000,000 when extending to 32K, helping 128K maximum during mid-training (Section 2.4; Figure 3).
  - Multi-Token Prediction `MTP`: an extra MoE head predicts several future tokens for fast speculative decoding (Section 2.1).
- Data pipeline (Sections 2.2–2.3; Figure 3)
  - Scale and sources: 23T tokens total. Web (English/Chinese heavy) with quality buckets; multilingual from FineWeb-2; curated code; math/science documents (Section 2.2).
  - Quality control:
    - Bucketing and up-sampling by quality scores; semantic deduplication (`SemDedup`) beyond MinHash to remove template-generated near-duplicates (Section 2.2).
    - Code triage into high/medium/low via per-language quality models; up-sample high, drop low; use Fill-In-the-Middle training for code (Section 2.2).
    - Two-stage pre-training: general web first; then up-sample code/math/science (Section 2.2).
  - Mid-training (Figure 3; Section 2.3)
    - `Repo-level code`: concatenate files and related GitHub issues/PRs/commits with diff formatting; extend sequences to 32K to learn cross-file context.
    - `Synthetic reasoning`: generate long reasoning traces for math/science/code competitions.
    - `Long-context & agent`: up to 128K contexts; inject synthetic agent trajectories.
    - `Best-fit packing` only in mid-training (to preserve long reasoning or repository spans); pre-training keeps random truncation for data augmentation (Section 2.3).
- Optimization and hyperparameters (Section 2.4)
  - `Muon` optimizer for all but embeddings/bias/RMSNorm; warmup to LR 2.5e-4 then cosine decay to 2.5e-5; batch-size warmup from 16M to 64M tokens; weight decay 0.1; no dropout.
  - Loss-free MoE routing bias updates: 0.001 rate for first 15T tokens, then 0.0; MTP loss weight 0.3 then 0.1 (Section 2.4).
- Post-training with “expert iteration” (Section 3)
  - Stage 1: Train separate expert models in Reasoning, Agent, and General chat with SFT + RL.
  - Stage 2: Unified SFT distills all experts into one generalist that can either “think” (extended chain-of-thought) or “answer directly” (Section 3.1).
    - Hybrid reasoning is enforced by mixing data with and without explicit reasoning traces so the model learns when to be concise (Section 3.1, “Overall SFT”).
- Data curation tricks for SFT (Section 3.1; Figure 4)
  - Function-call template: switch from JSON arguments to an XML-tagged schema that wraps keys/values, reducing escaping for code-heavy arguments while maintaining executability (Figure 4).
  - Rejection sampling: filter for formatting, non-truncation, correctness (objective tasks), reward-model scores (subjective), and valid tool-trajectory endpoints (Section 3.1).
  - Prompt selection + response scaling: discard the shortest 50% of prompts (by response length) and sample multiple responses for hard prompts (+2–4% on math/science; Section 3.1).
  - Automatic agentic SFT data: synthesize tools and tasks, roll out trajectories with LLMs, and retain only trajectories judged successful by multiple agents (Section 3.1).
- Reasoning RL (Section 3.2; Figures 5–7)
  - Base algorithm: GRPO-style objective without KL term; rewards from answer correctness or rubric checks (Section 3.2).
  - Difficulty curriculum: start with moderate problems, then switch to a pool of extremely hard problems with verified answers to keep reward variance informative (Figure 5).
  - Single-stage RL at full 64K output length: avoids “unlearning” long responses observed when doing shorter-length stages first (Figure 6).
  - Dynamic sampling temperature: increase temperature when rewards plateau, selecting the max temperature that keeps validation within 1% of the current best (Section 3.2).
  - Ablations:
    - Code RL: token-weighted mean loss (instead of sequence mean) speeds convergence and avoids length bias (Figure 7-left).
    - Science RL: small, expert-verified MCQ set beats larger mixed-quality data on GPQA-Diamond (Figure 7-right).
- Agentic RL (Sections 3.3.1–3.3.2; Figure 8)
  - Data synthesis for web search: multi-hop graph-based generation plus human-in-the-loop obfuscation of web content to create demanding, verifiable questions (Section 3.3.1).
  - Coding agents: GitHub PR/issue tasks with executable tests in a sandbox; scalable, isolated infra (Section 3.3.1).
  - Learning objective: group-wise policy optimization over K rollouts per prompt; reward is final answer correctness; add a format penalty that assigns zero reward if tool-call syntax is invalid, halting the trajectory (Section 3.3.2).
  - Iterative self-distillation: alternate RL and SFT using improved model outputs as new SFT targets to ratchet up performance (Section 3.3.2).
  - Test-time compute scaling: more browsing/interaction turns systematically increase BrowseComp accuracy (Figure 8).
- General RL (Section 3.4; Figure 9)
  - Holistic RL: balanced 5k-prompt set across 7→33→139 capability categories; rewards from a mixture of human preference models and rubric-based AI scoring to reduce bias (Section 3.4).
  - Instruction Following RL: taxonomy of 7 major/151 minor constraint types; deterministic checkers + reward model + critique model; SysBench-ISR score tracks rising reward without clear reward hacking up to 1k steps (Figure 9).
  - Function Calling RL:
    - Step-wise rule-based RL: strict reward only when the call matches the ground truth exactly and formatting is correct.
    - End-to-end multi-turn RL: reward is given only if the full trajectory is properly formatted and task-complete, with success judged by environment rules or an LLM judge (Section 3.4).
  - Pathology RL: targeted prompts to suppress rare but harmful behaviors (language mixing, repetition, formatting errors).
- RL infrastructure (Section 3.5; Figure 10)
  - Slime framework: supports colocated synchronous training for math/code RL and disaggregated asynchronous training for long-horizon agent tasks, with a `Data Buffer` decoupling rollouts and training (Figure 10).
  - Mixed precision: BF16 training + online FP8 quantized inference for faster rollouts per policy update (Section 3.5).
  - Agent-oriented design: high-concurrency Docker runtimes for isolated tasks; unified HTTP endpoint to ingest rollouts from heterogeneous agent frameworks; centralized pool with task-specific filters and dynamic sampling (Section 3.5).

## 4. Key Insights and Innovations
- Deeper-narrower MoE with many attention heads improves reasoning without hurting loss
  - What’s new: 96 heads at 5120 hidden dim plus QK-Norm and partial RoPE; fewer experts than some peers but more layers (Table 1).
  - Why it matters: Even though training loss does not improve, MMLU/BBH-style reasoning scores consistently do (Section 2.1), suggesting head count and depth can be tuned for reasoning.
- Long-context training aligned with RL procedures
  - What’s new: Train SFT and RL at the final 64K output length instead of staged length increases to prevent “unlearning” long responses (Figure 6).
  - Why it matters: Maintains 128K-usable behavior and keeps the model’s deliberation budget intact for hard tasks.
- Difficulty-calibrated Reasoning RL and data quality emphasis
  - What’s new: Two-stage curriculum switching to verified, extremely hard problems (Figure 5); token-weighted losses for code RL (Figure 7-left); expert-verified MCQs for science RL (Figure 7-right).
  - Why it matters: Keeps reward variance informative, speeds convergence, and prevents overfitting to noisy data.
- Hybrid-thinking generalist via expert iteration + distillation
  - What’s new: Train domain experts (Reasoning/Agent/General), then distill into one model that can either think step-by-step or answer concisely; training carefully balances CoT vs. no-CoT data (Section 3.1).
  - Why it matters: Enables fast responses on simple tasks and deliberate reasoning when needed; crucial for agentic reliability and user experience.
- Practical function-calling innovations
  - What’s new: XML-like call schema to avoid JSON escaping failures in code-heavy arguments (Figure 4); strict step-wise and trajectory-level rewards for formatting and correctness (Section 3.4).
  - Why it matters: Significantly reduces formatting/pathology errors that often derail real agent systems and improves tool-call success (Figure 13).
- Scalable agent RL infrastructure
  - What’s new: Asynchronous rollouts with FP8 inference; unified interfaces and centralized data pool for diverse agent frameworks (Section 3.5; Figure 10).
  - Why it matters: Makes it feasible to train on long, variable-length agent trajectories at scale.

## 5. Experimental Analysis
- Evaluation setup
  - Benchmarks cover agentic (TAU-bench retail/airline, BFCL V3 function calling, BrowseComp browsing), reasoning (MMLU-Pro, AIME 24, MATH-500, SciCode, GPQA, HLE, LCB), coding (SWE-bench Verified, Terminal-Bench), and general abilities (MMLU, SimpleQA, IFEval, SysBench, MultiChallenge) (Sections 4.2–4.2.4; Tables 3–6; Figure 1).
  - Base-model sanity check: GLM-4.5-Base (no instructions) is broadly competitive in English, Chinese, and code/math, showing stable pretraining quality (Table 2).
- Main numbers (selected highlights)
  - Agentic (Table 3)
    - TAU-bench: “retail” 79.7, “airline” 60.4; close to Claude Sonnet 4 (81.4, 60.0).
    - BFCL V3: 77.8, best among listed baselines.
    - BrowseComp: 26.4; behind o3 (49.7) and o4-mini(high) (28.3), ahead of Claude Opus 4 (18.8).
    - Average agentic score: 58.1 (2nd overall in Figure 1’s agentic panel).
  - Reasoning (Table 4)
    - AIME 24 Avg@32: 91.0 (beats o3 at 90.3; trails Grok 4 and Qwen3-235B-Thinking).
    - LCB: 72.9 (below o3 78.4; below Grok 4 81.9).
    - HLE: 14.4 (considerably below Gemini 2.5 Pro 21.1 and Grok 4 23.9).
    - AA-Index (estimated average across seven reasoning sets): 67.7 (close to DeepSeek-R1 68.3).
  - Coding (Table 5; Figure 2)
    - SWE-bench Verified: 64.2 (above GPT‑4.1 48.6, Gemini 2.5 Pro 49.0; close to Claude Sonnet 4 at 70.4).
    - Terminal-Bench: 37.5 (beats o3 30.2 and GPT-4.1 30.3; below Claude Sonnet 4 at 43.2).
    - Parameter efficiency: Figure 2 shows GLM‑4.5 and GLM‑4.5‑Air on the Pareto frontier of SWE-bench vs. parameter count.
  - General abilities (Table 6)
    - MMLU: 90.0 (comparable to top models).
    - IFEval: 86.1 (beats DeepSeek-R1 80.0).
    - SysBench: 81.0 (above GPT‑4.1 80.6; above DeepSeek V3 79.8).
    - MultiChallenge: 52.8 (beats GPT‑4.1 38.3; near Claude Sonnet 4 55.3).
  - Safety (Table 7)
    - Average SafetyBench: 89.9 (similar to GPT‑4.1’s 89.7; strong on Physical/Mental Health; room to improve on Unfairness & Bias at 77.4).
- Agentic coding hands-on (Section 4.3.2; Figures 12–13)
  - CC-Bench (52 real-world dev tasks; human-in-the-loop): GLM‑4.5 vs Qwen3‑Coder 80.8% win; vs Kimi K2 53.9% win; vs Claude Sonnet 4 splits 40.4%/9.6%/50.0% (win/tie/lose) (Figure 12).
  - Tool-call reliability: GLM‑4.5 has highest success rate (90.6%) with moderate token usage per interaction (Figure 13).
- Do the experiments support the claims?
  - The multi-benchmark picture shows strong all-around capability, especially for coding and structured tool use (Tables 3 and 5; Figures 12–13). Agentic browsing remains notably behind o3 (Table 3, BrowseComp).
  - Ablations and training curves credibly link methods to outcomes:
    - Difficulty curriculum (Figure 5), single-stage 64K RL (Figure 6), token-weighted loss (Figure 7-left), and expert-verified science data (Figure 7-right) all show measurable gains.
    - Instruction RL shows reward tracking SysBench-ISR without visible reward hacking up to 1,000 steps (Figure 9).
- Caveats in evaluation methodology
  - TAU-bench uses an “optimized user simulator” prompt (Figure 11), which can influence relative model performance; different simulators could change results.
  - Some judgments use LLM-based evaluators (e.g., HLE correctness by GPT‑4o; Section 4.2.2), which can introduce evaluator bias.
  - Human evaluations (Section 4.3.1) used a single evaluator per language batch; this avoids inter-rater variance but concentrates bias.
  - BrowseComp shows strong test-time scaling effects (Figure 8); models compared at one compute budget may differ at another.

> Examples of headline numbers:
> - “GLM‑4.5 scores 64.2% on SWE‑bench Verified and 37.5% on Terminal‑Bench” (Table 5).
> - “On AIME 24 (Avg@32), GLM‑4.5 reaches 91.0%” (Table 4).
> - “BFCL V3: 77.8, best among listed baselines” (Table 3).

## 6. Limitations and Trade-offs
- Performance asymmetries
  - Web browsing agents: GLM‑4.5 (26.4) lags far behind o3 (49.7) on BrowseComp (Table 3), suggesting weaker search/navigation strategies or evidence synthesis for web contexts.
  - Some reasoning sets (e.g., HLE 14.4) trail leading models (Table 4).
- Training complexity and compute
  - 23T tokens + long-context mid-training to 128K + multi-stage RL with large batch sizes implies heavy compute/data budgets (Sections 2–3).
  - MoE inference is efficient per token (32B active), but total parameter size (355B) still imposes infrastructure demands; speculative decoding via MTP helps but does not eliminate cost (Section 2.1).
- RL stability and reward design
  - GRPO without an explicit KL term can drift if reward models or rule-checkers mis-specify preferences; the paper mitigates this with validation temperature sweeps, strict format penalties, and pathology RL, but longer training horizons may still risk reward hacking (Sections 3.2–3.4; Figure 9).
- Data assumptions
  - Quality bucketing, SemDedup, and expert-verified pools are critical; effectiveness depends on the accuracy of these filters and classifiers (Section 2.2). Residual contamination or template biases could persist.
- Generalization vs. specialization
  - The hybrid model balances thinking and direct answering by data mixture; in edge cases, it may choose the wrong mode (overthink simple prompts or be too brief on hard ones). Although not quantified, this is an inherent trade-off in hybrid training (Section 3.1).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that an open-source MoE can approach proprietary systems across ARC tasks while being parameter-efficient (Figures 1–2; Tables 3–5).
  - Validates a pipeline that integrates: deeper-narrower MoE design, long-context training, expert iteration, difficulty-calibrated RL, and robust tool-call formatting and rewards.
- Research enabled
  - Architecture: Systematic studies on head count vs. depth for reasoning; interactions between QK-Norm, GQA, and partial RoPE at 128K context.
  - RL methodology: Further formalization of token-weighted objectives, difficulty curricula with verified pools, and temperature schedules tied to reward plateaus (Figures 5–7).
  - Agentic training: More realistic browsing environments; adaptive test-time compute policies to trade speed vs. accuracy (Figure 8).
  - Evaluation science: Standardized user simulators for TAU-like tests; multi-rater human eval protocols; LLM-judge consensus methods to reduce evaluator bias.
- Practical applications
  - Production agents: Function-calling assistants with fewer formatting failures (Figure 4) and higher tool-call success (Figure 13), especially for enterprise workflows (search, databases, tickets, terminals).
  - Software engineering: Repo-level fixes, test-driven development, and CI agents (Table 5; Section 3.3.1).
  - Education and analytics: Reasoning over math/science and long documents (AIME 24, MATH-500, LongBench lineage in references).
  - Translation and localization that require cultural/online-context reasoning, where GLM-4.5 outperforms specialized MT models on a curated difficult set (Table 12).

In sum, GLM‑4.5 contributes a practical, end-to-end recipe for training an open generalist that can think when needed, act reliably with tools, and code effectively. The paper’s ablations and infrastructure details make it a useful blueprint, while the mixed results on web browsing and some reasoning sets highlight fertile ground for next-stage improvements.
