# HOLISTICALLY EVALUATING THE ENVIRONMENTAL IMPACT OF CREATING LANGUAGE MODELS

**ArXiv:** [2503.05804](https://arxiv.org/abs/2503.05804)

## 🎯 Pitch

This paper offers the first comprehensive, data-driven assessment of the environmental footprint of building large language models, tracking not only the final training but also hardware manufacturing, model development, and water consumption across the entire lifecycle. By revealing that nearly half of total emissions and water use arise from the overlooked stages of model development and hardware, the work highlights critical blind spots in current AI sustainability reporting and provides essential data for more responsible policy and future model design.

---

## 1. Executive Summary
This paper provides a holistic, measurement-driven accounting of the environmental impact of building a family of language models (`OLMo`) across their full lifecycle: hardware manufacturing, model development (experimentation and hyperparameter tuning), final training, and simulated deployment (inference). Using sub-second power logging and region-specific energy and water factors, it finds total impacts of 493 metric tons of CO2e and 2.769 million liters of water for models up to 13B active parameters, with roughly half as much impact coming from development as from the final training runs.

## 2. Context and Motivation
- Problem/gap addressed
  - Most AI model reports disclose only the energy or emissions from a final training run, often assuming GPUs draw their maximum power continuously and ignoring development, manufacturing, and water use. Sections 2 and 3 argue this systematically underestimates impacts.
  - There is limited transparency on “embodied” impacts (from hardware manufacturing) and on water consumption across power generation and data center cooling (Sections 2, 3.2).

- Why this matters
  - Data center power demand is rapidly increasing, with projections of data centers using up to 11.7% of US electricity by 2030 (Introduction).
  - Environmental costs include both carbon emissions and water consumption along the supply chain (power plants, data centers, chip manufacturing). The paper quantifies both to inform responsible planning and policy (Abstract; Sections 3.1–3.2).

- Prior approaches and shortcomings
  - Prior reports typically:
    - Focus solely on final training (e.g., LLaMA/Llama 2/3, Gemma) and ignore development and inference (Section 2).
    - Assume GPUs run at 100% of theoretical max draw; this can misestimate energy (Section 1).
    - Rarely include water or embodied impacts due to lack of data (Section 2).
  - A few exceptions measure parts of this pipeline (BLOOM’s embodied emissions, some granular power logging), but none provide a complete accounting including development and water (Section 2).

- Positioning of this work
  - Extends best practices by measuring sub-second GPU power during development and training, calculating both CO2e and water across operational and embodied stages, and simulating inference with breakeven analyses (Sections 3–4).
  - Introduces grid-relevant analysis of intra-training power fluctuations (Section 4.3; Figure 2).

## 3. Technical Approach
This is an empirical measurement and estimation framework that spans multiple lifecycle stages.

- Stages covered (Sections 3.1–3.4; 4.1–4.2)
  1) Hardware manufacturing (“embodied” impacts).  
  2) Development (experiments before final training, captured as `H` in Equation 1).  
  3) Final training runs (`E·D` in Equation 1).  
  4) Simulated deployment/inference (downstream “Scope 3” usage).

- Conceptual framing (Equation 1; Section 3.1)
  - Training cost of a result `R` is proportional to `E·D·H`: per-example cost `E`, dataset size `D`, and number of experiments `H`. This paper explicitly measures both `E·D` (final training) and `H` (development).

- Operational impacts: how electricity, CO2e, and water are computed (Sections 3.1, 3.4)
  - Real-time measurement: GPU power is logged at sub-second intervals on one node and extrapolated to all nodes in a run. Only GPU power is measured, so estimates are lower bounds (Section 3.1).
  - Converting power to CO2e (Equation 2):  
    `CO2e = P · PUE · CI`  
    - `P`: total measured GPU energy (kWh or MWh).  
    - `PUE` (Power Usage Effectiveness): ratio of total data center energy to IT energy; captures cooling/overhead. Assumed 1.2 for `Jupiter` (Texas) and 1.12 for `Augusta` (Iowa) (Section 3.1; Section 4.1).  
    - `CI` (Carbon Intensity): kg CO2 per kWh on the local grid. 0.332 for Austin Energy (Texas); 0.352 for Iowa (Section 3.1).
  - Converting power to water consumption (Equation 3):  
    `Water = P · PUE · (WUE_onsite + WUE_offsite)`  
    - `WUE_onsite` (data center cooling) is 0 L/kWh due to closed-loop cooling (Section 3.1).  
    - `WUE_offsite` (water used in electricity generation) is 1.29 L/kWh for `Jupiter` and 3.10 L/kWh for `Augusta` (Section 3.1; Reig et al., 2020).

- Embodied impacts (hardware manufacturing; Section 3.2)
  - Embodied CO2e per 8×GPU server is assumed 3,700 kg CO2e; ≈463 kg per H100 GPU, amortized over a 4-year lifetime → ≈0.013 kg CO2e per GPU-hour (Section 4.1, “Hardware manufacturing”).  
  - Water per H100 is estimated at 100.4 L from published wafer-level estimates; plus 2.2 L for rare-earth mining. Amortized → ≈0.003 L per GPU-hour (Section 4.1, “Hardware manufacturing”).
  - Total GPU-hours used across the project: 1.65 million, yielding 22 tCO2e and 4.8 kL water from manufacturing (Section 4.1).

- Models, data, and hardware (Section 3.3)
  - Dense transformers from 20M to 13B active parameters; `OLMoE` is a mixture-of-experts with 1B active / 7B total parameters.  
    - “Active parameters” = the subset actually used per token (for MoE, only the routed experts participate).  
  - Training tokens:  
    - <1B: 1.7T tokens; `OLMo 1B`: 3T; `OLMo 7B`: 2–4T depending on variant; `OLMoE`: 5T; `OLMo 2 13B`: 5.6T (Figure 1; Section 3.3).
  - Hardware: 8×NVIDIA H100 per server, 2–128 nodes per run, high-speed interconnect (Section 3.3). All but the 13B trained on `Jupiter`; 13B trained on `Augusta` (Sections 3.1; 4.1).

- Simulated inference (Section 3.4; Table 3; Appendix Table 4)
  - Serving stack: SGLang on a single H100; requests sampled from the ShareGPT dataset (2,400 prompts).  
  - Traffic patterns: either batched (“∞”, arriving instantaneously) or Poisson arrivals at 8 req/s or 1 req/s; these map to realistic online serving regimes (Section 3.4).  
  - Measurement scope: tracks GPU energy for active inference processes using CodeCarbon; excludes CPU/RAM/idle overhead → lower bounds (Section 3.4; Table 3 note).  
  - Outputs: per-request energy/CO2e/water and “breakeven” number of inferences at which cumulative inference equals training CO2e (Table 3).

## 4. Key Insights and Innovations
- Holistic accounting across the full model lifecycle (fundamental)
  - The study quantifies manufacturing, development, final training, and simulated inference, and combines CO2e and water with sub-second measurement (Sections 3–4). Most prior work excludes development, water, or embodied impacts (Section 2).
  - Significance: provides a realistic, actionable footprint where common omissions can be as large as the reported training itself.

- Development is a first-class environmental cost (fundamental)
  - The paper is, to the best of the authors’ knowledge, the first to report detailed model development costs (Section 4.1; Table 1).  
  - Development emissions (159 tCO2e) are about half of final training emissions (312 tCO2e) and thus “∼50% of that of training” (Abstract; Section 4.1).

- Empirical evidence that GPU power is highly variable during training (novel measurement)
  - Sub-second logs show large, frequent drops in power during checkpointing: from over 600W (>85% of 700W TDP) down to about 100W (~15%) (Figure 2; Section 4.3).
  - Significance: this variability challenges power grid stability and invalidates common modeling assumptions that use constant 100% GPU draw (Abstract; Sections 4.3, 5.2).

- Inference “breakeven” analysis tied to serving regimes (practical innovation)
  - The breakeven number of inferences to match training emissions ranges from hundreds of millions to tens of billions depending on model and request rate (Table 3).  
  - Insight: Under-saturated serving (e.g., 1 req/s) yields similar per-request emissions across model sizes; peak efficiency gains from smaller models or batching don’t automatically translate to efficient deployment unless serving is optimized (Table 3 note).

- Water as a co-equal metric with CO2e (important extension)
  - The paper quantifies total water consumption (2.769 million liters) and shows it is acutely sensitive to data center location and electricity mix through `WUE_offsite` (Sections 3.1, 4.1; Table 2).

## 5. Experimental Analysis
- Evaluation methodology
  - No accuracy/benchmark comparison; the study evaluates environmental metrics throughout the lifecycle using measured GPU power, and converts to CO2e and water via Equations (2) and (3) with region- and site-specific factors (Section 3.1).
  - For inference, it measures GPU energy for serving 2,400 ShareGPT prompts in three request-rate regimes (Section 3.4; Table 3; Appendix Table 4).

- Main quantitative results (with sources)
  - Total project-wide impacts (Abstract; Section 4.1 “Putting it in perspective”):
    > “493 metric tons of carbon emissions … and consumed 2.769 million liters of water.”
    - Equivalents: energy use of ≈98 US homes for a year (EPA calculator) and ≈24.5 person-years of water use (Section 4.1).
  - By stage (Section 4.1; Table 1 and Table 2):
    - Manufacturing: 22 tCO2e; 4.8 kL water.
    - Development: 159 tCO2e; 843 kL water.  
      > “∼70% of our developmental environmental impact came from developing the 7B and 13B models.” (Table 1 narrative)
    - Final training: 312 tCO2e; 1,921 kL water (Table 2, “Total (Ours)”).
  - Training run specifics (Table 2):
    - `OLMo 2 13B` (5.6T tokens; trained on `Augusta` with higher `WUE_offsite`): 230 MWh, 101 tCO2e, 892 kL water.  
    - `OLMo 1B (3T)`: 30 MWh, 10 tCO2e, 39 kL water.  
    - `OLMoE 0924` (1B active, 5T tokens): 54 MWh, 18 tCO2e, 70 kL water.  
    - One `OLMo 7B` was trained entirely on hydropower (LUMI) and listed with 0 CO2e and 0 water for that run (Table 2, footnote).
  - Inference results and breakeven points (Table 3; Appendix Table 4):
    - Under saturated batching (“∞”), per-100-request time and per-request energy/CO2e are lowest. Example:  
      - `OLMo 2 7B` at “∞”: 0.018 kWh and 6.0 g CO2e per 100 requests; at 1 req/s: 0.358 kWh and 118.9 g CO2e.  
    - Breakeven inferences (CO2e parity with training):
      - `OLMo 2 7B`: 1.05B (1 req/s), 7.68B (8 req/s), 20.9B (“∞”).  
      - `Llama 2 13B`: 1.13B (1 req/s).  
      - `OLMo 1B (3T)`: 441M (1 req/s).
    - Observation highlighted in Table 3 note:
      > “relatively small variability in carbon emissions and water consumption across different model sizes in cases where batches are not saturated … greater peak efficiency does not guarantee efficient deployment if inference is not optimized.”
  - Power variability during training (Figure 2):
    > “When actively training, the average GPU power is over 600W … and during checkpointing, power usage drops to just over 100W.”

- Do the experiments support the claims?
  - The methodology carefully ties measured GPU energy to CO2e and water using site-specific `PUE`, `WUE_offsite`, and `CI` (Sections 3.1, 4.1). Sub-second power logs and per-run accounting across many models make the training/development results credible and reproducible in form.
  - Inference measurements are explicitly lower bounds and exclude server overhead (Table 3 note), but still suffice for comparative insights and breakeven orders of magnitude.
  - Embodied impacts rely on external estimates (Section 4.1), which the paper openly marks as assumptions.

- Ablations and robustness checks
  - Variation across model sizes (20M–13B active) and training tokens (1.7T–5.6T) shows how impacts scale (Figure 1; Tables 1–2).
  - Cross-datacenter runs (Texas vs Iowa; hydro for one 7B run) illustrate sensitivity to grid and `WUE` assumptions (Table 2).  
  - Real-time power traces expose system behavior around checkpointing (Figure 2).

- Conditions/trade-offs
  - Inference efficiency depends critically on request batching/saturation and thus on product workload characteristics (Table 3).  
  - Water use hinges on where you train (e.g., `Augusta`’s higher `WUE_offsite` drives large water for `OLMo 2 13B`) (Section 4.1; Table 2).

## 6. Limitations and Trade-offs
- Measurement scope and assumptions
  - GPU-only power logging excludes CPU, RAM, networking, and non-IT overheads; thus, operational energy/CO2e/water are lower bounds (Section 3.1; Table 3 note).
  - Embodied impacts are estimated from public reports; GPU vendors do not disclose official embodied CO2e and water, so results depend on third-party assumptions and amortization (Section 4.1 “Hardware manufacturing”; Section 5.1).
  - `WUE_offsite`, `PUE`, and `CI` are treated as constants per location/provider, which may vary over time or with marginal generation (Section 3.1).

- Inference simulations
  - Benchmarking is done on one H100 with a fixed software stack (SGLang) and specific traffic patterns; it excludes edge deployments, quantization variants, CPU-only scenarios, and diverse decoding/search settings (Appendix A.2; Section 3.4).
  - Only active GPU processes are measured, not background/idle costs of hosting (Table 3 note).

- System-specificity
  - Results pertain to two specific clusters (`Jupiter`, `Augusta`), their utilities, and model sizes up to 13B active parameters; scaling to much larger clusters or different geographies could change conclusions (Section 3.3; Section 4.1).

- Uncounted lifecycle stages
  - Transportation, data center construction, and end-of-life hardware disposal are acknowledged but not quantified (Section 4.1 “Other Costs”).

- Trade-offs surfaced by the study
  - Over-training smaller models (many more tokens) can reduce inference cost but raises training costs—risking a Jevons paradox where total consumption increases as efficiency improves (Section 5.2).
  - Frequent checkpointing increases resilience but causes large power oscillations that stress grids (Figure 2; Sections 4.3, 5.2).

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a template for full-lifecycle reporting: sub-second power logging, site-specific `PUE/CI/WUE`, explicit development accounting (`H`), embodied impact amortization, and inference breakeven analysis (Sections 3–4).
  - Shifts the conversation from “energy for the final training run” to a comprehensive view that includes manufacturing, development, and real-world serving patterns.

- Practical guidance for model developers and operators
  - Measure power at high frequency and include development; otherwise, you may miss roughly half of your training-phase footprint (Table 1 vs Table 2; Abstract).
  - Choose training locations with low `CI` and low `WUE_offsite`; closed-loop cooling (WUE_onsite ≈ 0) materially reduces water use (Section 3.1; Section 4.1).
  - Re-architect checkpointing and scheduling to smooth power demand (parallelized/streaming checkpoints, coordinated demand response) to reduce grid stress (Figure 2; Section 5.2).
  - For serving, optimize for saturation/throughput and consider “when/where” power is clean; otherwise, per-request emissions can be similar across model sizes despite peak-efficiency differences (Table 3).

- Policy and ecosystem implications
  - Standardized reporting of operational and embodied impacts should be required, including water; GPU manufacturers need to disclose embodied CO2e and water for transparency (Section 5.1).
  - Regulators can build on the EU AI Act/US proposals to define environmental disclosures that cover Scope 1–3 and water across manufacturing, operations, and end-of-life (Section 5.1).

- Research directions enabled/suggested
  - Better embodied impact data: vendor-verified life-cycle assessments for GPUs and data center construction (Section 5.1).
  - Energy- and water-aware training/inference schedulers that consider grid conditions, checkpoint timing, and multi-datacenter placement (Sections 4.3, 5.2).
  - Broader inference evaluations: edge devices, quantized models, alternative decoding, and realistic, unsaturated online traffic with full-system power (Appendix A.2; Section 3.4).
  - Methods to quantify and mitigate Jevons-like rebound effects when optimizing for deployment efficiency (Section 5.2).

> Bottom line: by rigorously measuring not just the final training runs but also development, manufacturing, and inference, the study shows that the overlooked parts of the lifecycle can be as large as—or larger than—the commonly reported numbers. The work provides concrete methods and actionable insights to reduce both carbon and water footprints while highlighting grid-level challenges that will grow with scale (Sections 3–5).
