# Holistically Evaluating the Environmental Impact of Creating Language Models

**ArXiv:** [2503.05804](https://arxiv.org/abs/2503.05804)
**Authors:** Jacob Morrison, Clara Na, Jared Fernandez, Tim Dettmers, Emma Strubell, Jesse Dodge
**Institutions:** 

## 🎯 Pitch

This paper presents the first comprehensive lifecycle analysis of open language models, quantifying often-overlooked environmental impacts such as water consumption and development phase emissions, alongside hardware manufacturing and training costs. By providing a holistic, measurement-based assessment, it reveals significant unreported environmental burdens and high intra-run power variability, emphasizing the need for transparent reporting and infrastructure optimization to mitigate the ecological footprint of AI systems.

---

## 1. Executive Summary
This paper quantifies, end to end, the environmental impact of creating a family of open language models (OLMo) by measuring electricity use, carbon emissions, and water consumption across hardware manufacturing, model development, final training, and simulated deployment. It shows that impacts commonly omitted from reports—especially model development and water use—are large: total emissions reach 493 metric tons CO2e and 2.769 million liters of water, and GPU power draw fluctuates sharply during training (≈15%–85% of max), which complicates grid planning (Abstract; §4.1; §4.3; Fig. 2).

## 2. Context and Motivation
- Problem addressed
  - Public reporting for foundation models usually covers only the last, successful training run and only CO2 from electricity; it rarely includes development runs, embodied impacts from hardware manufacturing, or water consumption, and almost never provides high‑resolution power data (§1; §2).
- Why it matters
  - Data centers could consume up to 11.7% of U.S. electricity by 2030; AI training and inference are major drivers (§1).
  - Without full accounting, stakeholders underestimate environmental costs, misplan grid capacity, and miss opportunities to reduce impacts (§1; §5).
- Gaps in prior work
  - Emissions often computed with simplifying assumptions such as “GPUs draw 100% of nameplate power” and without development or water accounting (e.g., Llama series reports; §2).
  - Embodied manufacturing impacts for GPUs are opaque; water use is underreported and often speculative (§2).
- Positioning
  - This work provides a holistic, measurement‑based life‑cycle view for a suite of LLMs (20M–13B active parameters; 1.7–5.6T tokens), including:
    - Operational impacts for development, training, and simulated inference (§3.1, §4.1–§4.2).
    - Embodied manufacturing impacts (§3.2).
    - High‑resolution (sub‑second) power traces revealing intra‑run variability (§3.1; §4.3).

Definitions (used throughout):
- `PUE` (Power Usage Effectiveness): How much extra facility power (cooling, overhead) is needed per unit of IT power; lower is better.
- `WUE` (Water Usage Effectiveness): Liters of water consumed per kWh; includes onsite cooling and offsite water used by power plants (§3.1; Eq. 3).
- Scope 2 emissions: Emissions from purchased electricity; Scope 3: upstream/downstream impacts, e.g., manufacturing and user‑side inference (not fully covered here, but partially via embodied and inference estimates; §3).

## 3. Technical Approach
The paper quantifies impacts using a modular pipeline that mirrors how models are actually created and used.

1) Operational impacts (development, training, inference)
- What is measured
  - GPU power draw sampled at sub‑second intervals for a representative node during each run, then extrapolated to all nodes in the job (§3.1).
  - The approach records only GPU power; CPU/network/storage overhead is not included, so operational numbers are lower bounds (§3.1).
- How electricity becomes CO2 and water
  - CO2 emissions:
    - Plain explanation: multiply electricity consumed by (a) how much extra facility power is needed and (b) the carbon intensity of the local grid.
    - Notation (Eq. 2): CO2e = P · PUE · CI, where `P` is IT power (kWh), `PUE` is a scalar (e.g., 1.2), and `CI` is kg CO2/kWh based on the local utility (§3.1).
  - Water consumption:
    - Plain explanation: each kWh causes water use at the data center (cooling) and at the power plant (steam/hydro losses).
    - Notation (Eq. 3): Water = P · PUE · (WUE_onsite + WUE_offsite). The paper’s data center uses closed‑loop cooling, so WUE_onsite = 0; offsite WUE is region‑dependent (§3.1).
- Where measurements were taken
  - Two H100 GPU clusters: “Jupiter” (Texas; Austin Energy: CI = 0.332 kg CO2/kWh; PUE = 1.2; WUE_offsite = 1.29 L/kWh) and “Augusta” (Iowa; CI = 0.351 kg CO2/kWh; PUE = 1.12; WUE_offsite = 3.10 L/kWh) (§3.1; §4.1).

2) Embodied manufacturing impacts
- Rationale: GPUs carry “upstream” embodied CO2 and water from manufacturing and materials extraction.
- Method
  - Adopt prior estimate of 3,700 kg CO2e per 8×GPU node (463 kg/GPU) from Luccioni et al. (2023) (§3.2; §4.1).
  - Water for fabrication approximated using TSMC process data: 12.33 L/cm² leading to ≈100.4 L per H100; add rare‑earth mining (assume 0.1% by mass), adding ≈2.2 L and 0.013 kg CO2e per GPU (§4.1 “Hardware manufacturing”).
  - Amortize over a 4‑year GPU life to get per‑GPU‑hour factors: ≈0.013 kg CO2e and 0.003 L water per GPU‑hour; multiply by total GPU‑hours used (§4.1).

3) Models, data, and hardware (empirical context)
- Dense transformers spanning 20M–13B active parameters; sub‑billion models trained on 1.7T tokens; 1B on 3T; several 7B variants on 2–4T; 13B on 5.6T (§3.3).
- One mixture‑of‑experts (MoE) model with 1B active/7B total parameters trained on 5T tokens (active = parameters used per token; MoE routes tokens to a subset of “experts”) (§3.3).
- Standard HGX servers with 8×NVIDIA H100; 2–128 nodes/run (§3.3).

4) Simulated deployment and inference
- Why simulate: the models were not deployed as a public service; to estimate downstream use, the authors emulate common chat usage (§3.4).
- Setup
  - Requests are sampled from the ShareGPT dataset (2,400 prompts) and fed to `SGLang` on a single H100 (§3.4).
  - Three arrival patterns: “batch all at once” (∞ req/s), 8 req/s, and 1 req/s; the latter two mimic Poisson arrivals used in online serving studies (§3.4).
  - Energy measured with CodeCarbon, cross‑checked against the same power logging used for training; only active GPU processes are counted (no idle/listening overhead) (§3.4; Table 3 note).

5) Power variability analysis
- Sub‑second GPU power traces show on‑off cycles during training, with sharp dips during checkpointing events (§4.3; Fig. 2).

## 4. Key Insights and Innovations
- First holistic accounting of LLM “creation” costs beyond the last training run
  - Novelty: separates and quantifies hardware manufacturing, development (ablation/tuning), the final training runs, and inference (§§3–4).
  - Significance: development alone emits ~50% as much CO2 as final training (159 vs. 312 tCO2e; Table 1 vs. Table 2), a large component usually unreported.
- Inclusion of water as a first‑class metric
  - Novelty: water consumption is computed both “onsite” (cooling; zero here due to closed‑loop) and “offsite” (from power generation), plus embodied water in hardware (§3.1–§3.2).
  - Significance: total water reaches 2.769 million liters when all phases are counted (§4.1 “Putting it in perspective”).
- High‑resolution measurement of non‑steady GPU power
  - Novelty: sub‑second traces show power swings from ≈85% of H100 TDP during training to ≈15% during checkpointing (Fig. 2); prior public reports typically assume constant draw (§4.3).
  - Significance: frequent, synchronized dips across many GPUs can destabilize grid operations and reduce efficiency (§5.2).
- Deployment-aware “break‑even” analysis
  - Novelty: for multiple models and serving regimes, the paper computes how many inferences are needed before inference CO2 equals training CO2 (§4.2; Table 3).
  - Significance: break‑even often occurs between hundreds of millions and tens of billions of inferences; production systems can reach this quickly (§4.2).

## 5. Experimental Analysis
- Evaluation methodology
  - Development and training
    - Runs grouped by model scale: <1B, 1B, 7B, 13B, and MoE (1B active/7B total). Each group includes many experiments for stabilization and hyperparameter sweeps before a final training run (§4.1; Table 1).
    - CO2 and water calculated from measured GPU power using Eq. (2) and Eq. (3), with data‑center‑specific PUE/WUE and local grid carbon intensities (§3.1; §4.1).
  - Embodied impacts
    - GPU manufacturing impacts amortized per GPU‑hour (details in §3.2; §4.1 “Hardware manufacturing”).
  - Inference
    - SGLang on 1×H100, 2,400 ShareGPT prompts, three arrival rates; power tracked by CodeCarbon; conversion to CO2/water uses the same PUE/WUE/CI as the training cluster (§3.4; Table 3 note).
- Main quantitative results
  - Development (Table 1)
    - Total: 680k GPU‑hours; 459 MWh; 159 tCO2e; 843 kL water.
    - Concentration: ~70% of development impact is from 7B and 13B scales (Table 1).
  - Final training runs (Table 2)
    - Total: 913 MWh; 312 tCO2e; 1,921 kL water.
    - Example runs:
      - `OLMo 2 13B` (5.6T tokens): 230 MWh, 101 tCO2e, 892 kL water.
      - `OLMo 2 7B` (4T tokens): 157 MWh, 52 tCO2e, 202 kL water.
      - `OLMoE 0924` (1B active/7B total, 5T tokens): 54 MWh, 18 tCO2e, 70 kL water.
      - One 7B model trained on a fully hydroelectric supercomputer (LUMI) records essentially zero operational CO2/water, illustrating location and energy‑mix sensitivity (Table 2, footnote “*”).
  - Whole‑program perspective (development + training + manufacturing)
    - Emissions: ≥493 tCO2e; Water: ≥2,769 kL (§4.1 “Putting it in perspective”).
    - Interpretation: “equivalent to … energy use for 98.2 U.S. homes in one year” and “24.5 years of water use by one average U.S. person” (§4.1).
  - Inference costs and break‑even (Table 3)
    - The table reports energy/CO2/water for the 2,400‑prompt benchmark (see table caption) and latency per 100 requests.
    - Examples at 1 req/s (unsaturated, realistic latency):
      - `OLMo 2 7B`: 0.358 kWh, 118.9 g CO2e, 0.533 L; 100.54 s per 100 req; break‑even ≈1.05 billion inferences.
      - `OLMo 1B (3T)`: 0.165 kWh, 54.8 g CO2e, 0.246 L; break‑even ≈441 million.
      - `Llama 2 13B`: 0.401 kWh, 133.1 g CO2e, 0.597 L; break‑even ≈1.13 billion.
    - Saturated throughput (∞ req/s) yields much lower per‑request energy than sparse arrivals, but many real applications cannot batch to that degree (Table 3; §4.2).
- Power variability (Fig. 2; §4.3)
  - >600 W per H100 when actively training (~85% of 700 W TDP) dropping to ~100 W during checkpointing (~15%); periodic dips reveal non‑steady demand.
- Are claims supported?
  - The training/development impact claims are strongly grounded in direct measurements and transparent conversions (Eq. 2–3; cluster‑specific PUE/CI/WUE).
  - Embodied impacts are necessarily approximate due to industry opacity (explicitly acknowledged in §5.1 “Embodied emissions are still an enigma”).
  - Inference results are careful lower bounds (GPU only; no CPU/host overhead), with caveats clearly stated (Table 3 note; Appendix A.2).

> Table 1 shows development emits 159 tCO2e and consumes 843 kL water, while Table 2 shows 312 tCO2e and 1,921 kL for final training runs—i.e., development ≈50% of training in CO2 and ≈44% in water.

> Figure 2 shows GPU power cycling between ~85% and ~15% of max due to checkpointing, indicating significant intra‑run variability that typical “100% power” assumptions miss.

## 6. Limitations and Trade-offs
- Measurement scope
  - GPU‑only power: excludes CPU, memory, networking, and idle/listening overhead; operational impacts are lower bounds (§3.1; Table 3 note).
  - Single‑node sampling extrapolated to whole job; if nodes are heterogeneous or desynchronized, extrapolation error can grow (§3.1).
- Embodied impacts uncertainty
  - Manufacturing CO2 and water rely on secondary sources and assumptions (e.g., 463 kg CO2e/GPU; 100.4 L water/GPU; 0.1% rare‑earth mass fraction), amortized over a 4‑year life (§4.1 “Hardware manufacturing”).
  - Other Scope 3 elements (transport, data‑center construction, end‑of‑life) are not fully included (§4.1 “Other Costs”).
- Inference realism
  - Simulations use a single H100 with SGLang on 2,400 ShareGPT prompts; results do not capture diverse serving stacks, quantization, edge deployment, multi‑GPU inference, or interactive/streaming patterns (§3.4; Appendix A.2).
  - Measured values exclude system overhead, leading to optimistic per‑request energy (§3.4; Table 3 note).
- Generalizability
  - Results depend on region (grid carbon intensity, offsite WUE), data‑center efficiency (PUE), and training recipes (e.g., checkpoint frequency), limiting direct transfer to other sites (§3.1; Table 2).
  - Observed near‑linear training cost trends with model size may not hold in decentralized or multi‑datacenter training with higher communication overhead (Appendix A.2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Moves environmental reporting from “final training CO2 only” toward a lifecycle assessment that includes development, water, and power dynamics. This raises the bar for transparency and provides concrete numbers to guide procurement, scheduling, and model design (§5.1–§5.2).
- Practical takeaways
  - Track and publish development costs: development can be ≈50% of training in CO2 (Table 1 vs. Table 2), so optimizing early-stage experiments (e.g., simulation, better scaling‑law ladders) can yield large savings (§4.1; §5.1).
  - Choose location and timing strategically: switching regions or energy sources can drive impacts near zero (e.g., 7B on LUMI hydro power in Table 2). Demand‑response scheduling could also reduce marginal grid intensity (§5.2).
  - Reduce power variability: parallelized/asynchronous checkpointing and better job orchestration can minimize high‑frequency load swings that strain grids (Fig. 2; §5.2).
  - Account for water: offsite WUE dominates in closed‑loop‑cooled facilities; grid mix (e.g., thermoelectric vs. wind/solar) materially affects water use (§3.1; §5.1).
  - Deployment optimization matters as much as training efficiency: break‑even can be reached in 10^8–10^10 inferences; unsaturated serving (1 req/s) is far less efficient than batched scenarios (Table 3; §4.2).
- Policy and reporting
  - The analysis underscores the need for standardized, auditable disclosures (e.g., EU AI Act reporting; U.S. Artificial Intelligence Environmental Impacts Act proposals) and for GPU manufacturers to publish embodied impact factors (§5.1).
- Research directions
  - Better embodied-impact models for chips and systems; public datasets of per‑component CO2/water.
  - Scheduling and systems research to smooth power demand (e.g., checkpoint‑aware schedulers, grid‑integrated datacenters).
  - Methods to estimate real‑world inference footprints, including system overhead, edge devices, and mixed workloads.
  - Training pipelines that minimize development runs (e.g., transfer‑efficient methods, compute‑efficient scaling‑law ladders; see §3.1 Eq. 1 and §5.1).

In sum, this paper provides a rigorous, transparent template for holistic environmental accounting of LLMs, reveals substantial unreported costs (development and water), and identifies systems‑level issues (power variability) that require attention from both ML practitioners and infrastructure planners.
