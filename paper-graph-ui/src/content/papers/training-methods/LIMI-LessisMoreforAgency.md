# LIMI: Less is More for Agency

**ArXiv:** [2509.17567](https://arxiv.org/abs/2509.17567)

## 🎯 Pitch

LIMI ('Less Is More for Agency') fundamentally challenges the prevailing belief that large-scale data is required to cultivate sophisticated autonomous AI agents. By strategically curating just 78 high-quality agentic demonstrations, LIMI enables a large language model to outperform state-of-the-art baselines trained on datasets up to 128 times larger, achieving remarkable agentic intelligence on real-world collaborative tasks. This breakthrough reveals that the essence and quality of demonstrations, not sheer data volume, are the key to developing practical, autonomous 'working AI' for complex environments—a finding poised to reshape the principles of building truly agentic AI systems.

---

## 1. Executive Summary

This paper introduces LIMI (Less Is More for Intelligent Agency), a data-centric approach demonstrating that sophisticated agentic capabilities can emerge from minimal but strategically curated training demonstrations rather than from large-scale data accumulation. Using only 78 carefully designed training samples focused on collaborative software development ("vibe coding") and scientific research workflows, LIMI—a fine-tuned GLM-4.5 model—achieves 73.5% on the AgencyBench benchmark, dramatically outperforming state-of-the-art models including GLM-4.5 (45.1%), Kimi-K2-Instruct (24.1%), and DeepSeek-V3.1 (11.9%). The paper establishes the **Agency Efficiency Principle**: machine autonomy emerges not from data abundance but from strategic curation of high-quality agentic demonstrations, with LIMI achieving 53.7% improvement over models trained on 10,000 samples—128× fewer training examples—establishing that agentic intelligence follows fundamentally different development principles from traditional scaling laws, where data efficiency rather than data volume governs the emergence of autonomous task execution, multi-step reasoning, and collaborative problem-solving capabilities.

## 2. Context and Motivation

### The Core Problem: We Don't Know How to Cultivate Agentic Intelligence Efficiently

The fundamental question this paper tackles is deceptively simple: **does developing autonomous AI agents—systems that can discover problems, formulate hypotheses, and execute solutions through self-directed engagement with tools and environments—truly require exposure to vast amounts of training data, or can these capabilities emerge through more strategic, data-efficient approaches?** This matters because the field of agentic AI is currently following the same brute-force scaling paradigm that dominated language model pretraining: collect more data, train on larger datasets, and hope that agentic behaviors emerge as an implicit byproduct of scale.

The paper identifies a critical gap in our understanding. While traditional language model scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) have established clear relationships between model size, data volume, and performance on next-token prediction, **no equivalent framework exists for agentic intelligence**. The field has largely assumed—without rigorous empirical testing—that more training data yields better agency, following the same patterns observed in language modeling pretraining. As the authors state in Section 1:

> "Current approaches assume that more data yields better agentic intelligence, following traditional scaling laws from language modeling... yet this fundamental assumption remains largely untested: do agentic capabilities truly require exposure to vast amounts of training data, or could they emerge more efficiently through strategic approaches?"

This gap is significant for several practical and theoretical reasons that the paper highlights:

**The economic imperative.** Training state-of-the-art language models already requires enormous computational resources, and adding agentic capabilities on top—through large-scale data collection of interaction trajectories, tool-use demonstrations, and multi-turn collaborative sequences—threatens to compound these costs dramatically. If agentic intelligence can be cultivated with orders of magnitude less data than currently assumed, it fundamentally changes the economics of building autonomous AI systems, making sophisticated agency accessible to organizations without hyperscaler-level compute budgets.

**The capability bottleneck in industry adoption.** The introduction frames this through a stark distinction: "industries demand autonomous agents that can execute tasks, operate tools, and drive real-world outcomes" but current AI excels at "reasoning and generating responses" without demonstrating reliable autonomous execution. This gap between thinking AI and working AI represents perhaps the most significant barrier to deploying LLMs in production settings. An AI that can explain how to fix a bug is useful; an AI that autonomously navigates a codebase, diagnoses the issue, implements a fix, and verifies the solution is transformative. Bridging this gap efficiently is of enormous practical value.

**The scientific puzzle of emergent agency.** Beyond practical concerns, there is a deeper theoretical question: is agentic intelligence fundamentally different from language modeling capability? If agency is simply an emergent property of scale—more parameters, more data, more training compute—then the path forward is clear but expensive. If, however, agency follows fundamentally different development principles, as the paper hypothesizes, then understanding these principles becomes essential to guiding the field's research direction. The paper's title itself—"Less Is More for Agency"—positions this as the central intellectual contribution.

### Conflicting Signals from Adjacent Domains

The paper is motivated by a genuine tension in recent research findings. On one side, the dominant paradigm in agentic AI development has been data-intensive. State-of-the-art agentic models like GLM-4.5 (Zeng et al., 2025) and Kimi-K2 (Team et al., 2025) are trained on massive datasets, employing "large-scale data synthesis and extensive computational resources" (Section 5.1). The natural assumption—reinforced by the success of scaling laws in language modeling—has been that more interaction trajectories, more tool-use demonstrations, and more collaborative scenarios will produce better agents.

On the other side, emerging evidence from adjacent domains suggests a compelling alternative. The paper explicitly cites two key precedents:

- **LIMA** (Zhou et al., 2023) demonstrated that only 1,000 carefully curated prompt-response pairs can achieve effective model alignment, with the resulting model generalizing across diverse tasks despite its minimal training set. The core insight was that alignment quality depends on demonstration quality, not quantity—a finding that directly challenged the prevailing assumption that alignment requires large-scale instruction tuning datasets.

- **LIMO** (Ye et al., 2025) extended this paradigm to complex mathematical reasoning, achieving a 45.8% absolute improvement with only 817 strategically selected training samples—approximately 1% of the data typically required for comparable reasoning performance. The finding suggested that sophisticated cognitive capabilities like mathematical reasoning can emerge from minimal but carefully constructed demonstrations, rather than requiring exposure to thousands of examples.

These convergent findings from language alignment and mathematical reasoning create a natural but unexplored question: does agentic intelligence follow similar efficiency principles? The paper's contribution is to extend the Less-Is-More paradigm into the domain of autonomous agency, where the capabilities in question—tool orchestration, multi-step reasoning across extended interaction sequences, collaborative communication, strategic planning with environmental feedback—are substantially more complex than single-turn language generation or mathematical problem-solving.

### Where Existing Approaches Fall Short

The paper identifies specific limitations in current approaches to developing agentic AI systems along several axes:

**1. The untested assumption of data scaling.** The most fundamental shortcoming is that the data-scaling hypothesis for agency has not been empirically validated. Current training methodologies in leading agentic models "predominantly rely on large-scale data synthesis and extensive computational resources" (Section 5.1), following an implicit assumption that more interaction data produces better agents. The paper does not cite any prior work that systematically compares small, curated agentic training sets against large-scale alternatives at equivalent model scales. This means the field has been investing enormous resources in data collection and training pipelines without knowing whether a more efficient path exists.

**2. The gap between language capability and autonomous execution.** The introduction draws a sharp distinction between models that think and models that work. Current LLMs demonstrate impressive reasoning capabilities when prompted—they can explain problem-solving strategies, generate step-by-step plans, and reason about hypothetical scenarios. However, as the paper notes, "industries demand autonomous agents that can execute tasks, operate tools, and drive real-world outcomes." The jump from generating coherent text about how to solve a problem to autonomously executing the solution through tool interactions, environmental feedback loops, and adaptive strategy adjustment is not automatic. It requires specific training on agentic behaviors—but the question is how much and what kind of training data is sufficient.

**3. The lack of frameworks for agentic intelligence development.** Unlike language modeling, where scaling laws provide clear guidance about how to allocate pretraining compute, or alignment, where instruction-tuning recipes are well-established, agentic AI development lacks a principled framework. The paper positions LIMI as providing such a framework through its three core innovations (Section 1): query synthesis methodologies that capture authentic agentic patterns, systematic trajectory collection protocols that record complete interaction sequences, and the data efficiency principle itself, which establishes that curation quality matters more than dataset size.

**4. The absence of benchmarks targeting real-world agency.** Prior to AgencyBench (Li et al., 2025b), the agentic AI evaluation landscape was fragmented across narrow capability assessments: tool-use benchmarks like τ-bench (Yao et al., 2024a) test specific interaction patterns, code generation benchmarks like HumanEval (Liu et al., 2023) measure isolated programming ability, and scientific computing benchmarks like SciCode (Tian et al., 2024) evaluate domain-specific computation. None of these capture the integrated, multi-turn, collaborative nature of real-world agentic workflows that span planning, execution, tool orchestration, and adaptive problem-solving within a single coherent task. The paper's evaluation framework directly addresses this gap by using AgencyBench as a primary benchmark while also testing generalization across established benchmarks to validate that LIMI's improvements are not benchmark-specific.

### How This Paper Positions Itself

The paper positions itself as a **paradigm challenge**, not a methodological refinement. Rather than proposing a new training algorithm, architecture, or data augmentation technique, it argues that the fundamental assumption driving agentic AI development—that more data is better—is wrong for this specific capability. The contribution is therefore conceptual and empirical rather than algorithmic: the paper demonstrates that **strategic data curation yields superior agentic intelligence compared to large-scale data accumulation**, and does so with a margin (53.7% improvement over 10,000-sample training with only 78 samples) that makes the finding difficult to dismiss as a minor efficiency tweak.

The paper explicitly draws a lineage from LIMA and LIMO, positioning itself as the third demonstration of the Less-Is-More principle, applied to a fundamentally more complex capability. The narrative arc is: alignment requires only 1,000 examples (LIMA), mathematical reasoning requires only 817 examples (LIMO), and now agentic intelligence—which encompasses planning, tool use, multi-step execution, and collaborative interaction—requires only 78 examples (LIMI). The decreasing sample count across increasingly complex capabilities is rhetorically powerful and suggests a pattern: the more sophisticated the target behavior, the more important demonstration quality becomes relative to quantity.

Crucially, the paper does not claim that LIMI achieves *universal* agency—it focuses on two domains (vibe coding and research workflows) that "collectively span the majority of knowledge work scenarios" (Section 2.2). This scoping is important: the claim is that within these domains, strategic curation beats scale, not that 78 examples suffice for arbitrary agentic tasks. The paper also does not claim that pretraining scale is irrelevant—LIMI is built by fine-tuning GLM-4.5, a large pretrained model. The claim is specifically about the **training data needed to elicit agentic capabilities from an already-capable base model**, analogous to how LIMA showed that alignment requires quality over quantity for instruction tuning.

### The Real-World Stakes

The paper's motivation is not purely academic. It frames the transition from thinking AI to working AI as "the dawn of the Age of AI Agency" (Section 1), driven by an "urgent industry shift." This framing connects the technical contribution to a broader narrative: as AI systems become more capable of reasoning, the next frontier is autonomous execution, and whoever can cultivate this capability most efficiently gains a significant advantage. If LIMI's findings generalize—if agentic intelligence indeed follows fundamentally different development principles from language modeling—then organizations that invest in strategic data curation will outpace those that simply scale up data collection pipelines.

The paper also implicitly addresses a sustainability concern. The scaling paradigm leads to "increasingly complex training pipelines and substantial resource requirements" (Section 1). By demonstrating that 78 carefully curated examples can outperform 10,000 less curated ones, LIMI suggests a path toward more sustainable development of agentic AI—one where the quality of understanding about what constitutes effective agency matters more than the quantity of compute thrown at data collection.

## 3. Technical Approach

### 3.1 Reader Orientation

LIMI is a fine-tuned large language model that has been trained on a small, strategically curated set of 78 human-AI collaborative interaction trajectories—essentially transcripts of an expert AI agent working alongside a human to solve complex software development and scientific research tasks. The problem it addresses is the inefficiency of current agentic AI training: existing approaches assume that throwing more training data at models produces better autonomous agents, but LIMI demonstrates that the quality and authenticity of the agentic demonstrations matter far more than their quantity—78 carefully chosen examples can yield an agent that dramatically outperforms models trained on up to 10,000 less-curated trajectories, achieving 73.5% on the comprehensive AgencyBench benchmark where the next-best model reaches only 45.1%.

### 3.2 Big-Picture Architecture (Diagram in Words)

The LIMI system has five major components connected in a data pipeline that flows from query creation through trajectory collection to model fine-tuning and evaluation:

1. **Query Pool Construction**: A set of 78 real-world collaborative tasks is assembled from two sources—genuine queries encountered by professional developers and researchers (60 queries), and queries synthesized from GitHub Pull Requests using GPT-5 (18 queries, selected from thousands of candidates through expert review). Each query is a natural language specification of a multi-step collaborative objective, such as "Build a Gomoku game with progressively sophisticated AI opponents" or "Search Hugging Face for datasets matching specific criteria and extract their metadata."

2. **Execution Environment (SII CLI)**: A command-line interface environment equipped with integrated tools for software development (code editing, compilation, testing), research activities (data analysis, literature search), and information processing. The SII CLI serves as the "world" in which agentic interactions occur—it provides the tools the agent can invoke, captures detailed logs of all interactions, and enables the human collaborator to provide feedback and guidance.

3. **Trajectory Collection Protocol**: For each of the 78 queries, a PhD student annotator serves as a human collaborator working with GPT-5 (acting as the agentic model) within the SII CLI environment. The interaction continues iteratively—through cycles of model reasoning, tool invocation, environmental feedback, and human guidance—until the task is successfully completed. The complete sequence of actions is recorded as a trajectory, capturing authentic patterns of agentic behavior including error recovery, strategy adaptation, and collaborative communication.

4. **Training Dataset Assembly**: The collected trajectories are formatted as supervised fine-tuning data \((q_i, \tau_i)\) where \(q_i\) is the initial user query and \(\tau_i = \{a_{i,1}, \ldots, a_{i,n_i}\}\) is the sequential interaction history showing how an expert agent solved the problem. Each action in the trajectory represents one of three types: model reasoning output, model tool invocation, or environmental observation (including human feedback).

5. **Fine-Tuning Module**: Using the Slime framework, the base model (GLM-4.5 or GLM-4.5-Air) is fine-tuned via standard supervised learning on these 78 trajectories. The fine-tuned model—now called LIMI—learns to replicate the expert agentic behavior patterns demonstrated in the training data, including how to reason about complex tasks, orchestrate multiple tools, respond to environmental feedback, and collaborate with human partners.

Information flows as follows: real-world development scenarios and GitHub PRs → GPT-5 synthesizes structured queries → PhD annotators review and select the highest-quality queries → annotators collaborate with GPT-5 in SII CLI to solve each query → the complete interaction sequences are recorded as trajectories → these 78 trajectories form the training dataset → GLM-4.5 is fine-tuned on this curated data → LIMI is evaluated on AgencyBench and generalization benchmarks.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal definition of queries and trajectories that establishes the mathematical structure of LIMI's training data, since this notation underpins everything that follows.
- **Second**, the query pool construction methodology, covering both real-world query collection and the GitHub PR-based synthesis pipeline, to understand what makes these 78 queries special and why they capture authentic agentic patterns.
- **Third**, the trajectory collection protocol within the SII CLI environment, explaining how interactions are structured, what information each trajectory captures, and why the iterative-until-success approach matters for data quality.
- **Fourth**, the training procedure and model variants, including the Slime framework, the comparative datasets used for data efficiency experiments, and the evaluation configuration with and without CLI access.
- **Fifth**, the evaluation framework across AgencyBench and generalization benchmarks, covering the specific metrics (FTFC, SR@R, RC@R) and what they measure about agentic capability.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **data-centric empirical paper** whose core idea is that agentic intelligence can be cultivated through strategic curation of a small number of high-quality training demonstrations rather than through large-scale data accumulation, and that the key to effective data curation lies in capturing authentic, complete trajectories of expert human-AI collaborative problem-solving.

---

#### Formal Framework for Agentic Interaction

The paper formalizes the training data construction process around a tuple structure that cleanly separates the initial task specification from the subsequent interaction dynamics. This formalization is essential because it establishes what constitutes a "training example" for agentic intelligence and defines the data that the model eventually learns from.

A **query** \(q_i\) is the initial natural language specification provided by a user that articulates the desired collaborative objective. The query "establishes both the starting point and success criteria for the subsequent collaborative process" (Section 3.1). It is not merely a prompt asking for information—it is a work order that initiates an extended collaborative workflow, such as "Build a C++ console chat system with escalating features like user registration, friend management, chat history, and concurrency support" (Task 1 from AgencyBench, Appendix A.1).

A **trajectory** \(\tau_i = \{a_{i,1}, \ldots, a_{i,n_i}\}\) captures the complete interaction sequence that follows from the initial query \(q_i\) until successful task completion. The trajectory is an ordered list of actions where the index \(j\) maintains temporal ordering, and \(n_i\) represents the total number of actions required for query \(i\) to reach resolution.

Each action \(a_{i,j}\) in the trajectory is one of three fundamental interaction types:

- **Model reasoning** \(m_{i,j}\): The agentic model's cognitive output, which captures "understanding, analysis, planning, and decision-making processes" (Section 3.1). This is what the model thinks before acting—its internal monologue about what sub-problem to tackle next, what strategy to employ, or how to interpret previous feedback. In the actual training data, this corresponds to the reasoning text the model generates between tool invocations.

- **Model tool calling** \(t_{i,j}\): Structured tool invocations executed by the model to interact with external environments and accomplish specific subtasks. These are concrete actions rather than thoughts—running a compilation command, writing code to a file, searching a dataset repository, or executing a data analysis script. Each tool call has a specific format that the execution environment can parse and execute.

- **Environment observation** \(o_{i,j}\): Results and outputs returned from tool executions, as well as user feedback and clarifications provided during the collaborative process. These observations "inform subsequent model reasoning cycles" (Section 3.1), closing the perception-action loop that characterizes agentic behavior. Without environment observations, the model would have no way to know whether its tool invocation succeeded, what the results were, or what adjustments to make next.

The triple \((m_{i,j}, t_{i,j}, o_{i,j})\) is not necessarily present at every step—some steps may consist solely of reasoning without tool invocation, while others consist of tool invocation followed by observation. The formalization groups these under the unified action type \(a_{i,j}\) to maintain a clean sequential structure while allowing flexibility in the actual interaction pattern.

This formalization captures the complete collaborative workflow by separating the initial task specification \(q_i\) from the dynamic interaction process \(\tau_i\). The separation matters because it ensures the model learns both how to interpret initial task specifications (mapping from \(q_i\) to the first reasoning step) and how to maintain coherent, goal-directed behavior across extended interaction sequences (processing each subsequent observation and deciding the next action). The model cannot succeed by simply memorizing query-response pairs—it must learn the full dynamic of agentic interaction, including error recovery, strategy adaptation, and collaborative communication.

The paper notes that the longest collected trajectory reaches 152,000 tokens, with an average trajectory length of 42,400 tokens (Figure 4). These figures demonstrate that the 78 training examples represent far more than 78 simple prompt-response pairs—they are extended, multi-turn interaction sequences that collectively encode thousands of individual agentic decision points. This explains how 78 examples can provide sufficient learning signal: each example is dense with demonstrations of planning, tool orchestration, environmental interpretation, and adaptive problem-solving.

---

#### Query Pool Construction

The query pool construction is the first stage of the LIMI pipeline and perhaps the most critical, since the entire approach rests on the quality and authenticity of the queries that initiate agentic trajectories. Without queries that genuinely require the full spectrum of agentic capabilities—planning, tool orchestration, multi-step reasoning, collaborative interaction—the collected trajectories would not capture the patterns needed for effective learning. The paper employs a two-pronged strategy: collecting queries from real-world professional practice and synthesizing additional queries from open-source software development artifacts.

**Real-World Query Collection: 60 Queries**

The first source comprises 60 queries "collected from actual scenarios encountered by professional developers and researchers in collaborative environments" (Section 3.2). These are not artificial benchmarks or synthetic constructions—they represent genuine challenges that arose in the course of real work. The queries span both target domains: vibe coding (collaborative software development tasks) and research workflows (scientific investigation tasks involving data analysis, literature search, experiment design, and insight generation).

A notable methodological detail: "a substantial portion of the research queries are derived from real academic papers (Xiao et al., 2025a,b; Jiang et al., 2025; Li et al., 2025a; Sun et al., 2025, 2024)" (Section 3.2). This is an important claim about ecological validity—the research queries are not simplified textbook problems but rather tasks extracted from actual published research, ensuring that the training data captures "authentic representation of genuine research challenges." The connection to specific papers implies that these queries involve the kind of complex, multi-step investigative work that characterizes real academic research: comparing model performance across conditions, analyzing statistical patterns, searching for and extracting metadata from research datasets, and fitting mathematical models to scientific data.

The 60 real-world queries provide the backbone of the training data's authenticity. Because they originate from actual professional practice, they naturally capture the complexity, ambiguity, and contextual richness of real collaborative work—elements that would be difficult to specify a priori in a synthesis framework. However, 60 queries is a small number, especially when split across two domains. This motivates the second source.

**GitHub PR-Based Query Synthesis: 18 Additional Queries**

To systematically expand the query pool while maintaining authenticity, the paper develops a pipeline for synthesizing queries from GitHub Pull Requests using GPT-5. The intuition is that PRs represent concrete, real-world software development tasks—they document specific code changes, bug fixes, feature additions, and optimizations that actually occurred in production codebases. By transforming these concrete code changes into forward-looking collaborative development tasks, the pipeline generates queries that reflect genuine development needs while expanding coverage beyond what the 60 real-world queries captured.

The synthesis pipeline involves five stages, each designed to filter for quality and relevance:

**Stage 1: Repository Selection.** The pipeline selects repositories with more than 10,000 GitHub stars. This threshold serves as a heuristic for codebase quality and community significance—high-star repositories tend to have well-maintained code, clear documentation, and substantive pull requests that reflect meaningful development activity. The paper does not specify the exact number of repositories that passed this filter, but it states that "100 repositories" were ultimately selected after the domain diversification step.

**Stage 2: Domain Diversification.** From the high-star-repository pool, the authors ensure "comprehensive coverage across diverse software development domains, including frontend development, backend systems, deployment infrastructure, debugging, and code optimization, among others" (Section 3.2). The final selection of 100 repositories is designed to span the range of software development activities that an agentic AI might encounter in practice. This diversification is important because it prevents the synthesized queries from being narrowly focused on a single type of development task (e.g., all frontend UI changes), which would limit the agentic patterns captured in subsequent trajectories.

**Stage 3: Complexity Filtering.** From the 100 selected repositories, the pipeline collects 1,000 PRs per repository (100,000 PRs total), then applies two filters. First, only PRs with a "unified diff patch token count below 1,200 tokens" are retained—this excludes massive refactoring PRs that would be too complex for a single coherent query while keeping PRs that represent focused, meaningful code changes. Second, PRs that "only modify Markdown files" are excluded, ensuring that the pipeline focuses on substantive code changes requiring meaningful agentic intervention. These filters are pragmatic: they aim for PRs that are complex enough to require genuine agentic capabilities (planning, tool use, understanding of codebase context) but focused enough to be expressed as coherent, actionable queries.

**Stage 4: Sampling for Synthesis.** From the filtered pool, 100 PRs are randomly sampled per repository for query synthesis. With 100 repositories, this yields up to 10,000 candidate queries—but the actual number synthesized depends on the quality assessment in the next stage. The random sampling ensures statistical representativeness while controlling annotation workload.

**Stage 5: Quality Assurance and Final Selection.** Four PhD students in computer science serve as expert annotators to evaluate the quality of synthesized queries. The evaluation criteria focus on "semantic alignment between the generated query and the corresponding PR content" (Section 3.2)—does the synthesized query accurately capture the intent and context of the real development scenario? This human-in-the-loop quality assessment is essential because GPT-5's synthesis might produce queries that are superficially plausible but fail to capture the nuanced technical requirements, constraints, or context that make the PR a meaningful development task.

The prompt used for query synthesis is provided in Appendix B of the paper. The prompt instructs GPT-5 to analyze the PR data (title, description, file changes, commits, discussion context), identify the primary task category from a predefined taxonomy (algorithm development, application development, LLM development, backend development, UI optimization, frontend development, build/deployment, research, data processing, or debugging), extract key information (repository ID, PR number, modified files, related files, specific technical requirements), and generate a "clear, specific query that an AI agent could receive to implement this task." The generated query must be "written as if from a developer requesting help," include "specific technical requirements," mention "key technologies, frameworks, or patterns involved," and be "concrete enough to have a testable outcome" (Appendix B).

Through this systematic process, the pipeline generates "several thousand high-quality synthetic queries" (Section 3.2). However, the final dataset for LIMI uses only 18 such queries. The selection criterion is alignment with the two core domains: the authors "strategically sample 18 queries that best match" vibe coding and research workflows. This means that only a small fraction (roughly 0.2–0.5%) of the synthesized queries make it into the LIMI training set. The remaining synthesized queries and their trajectory data are reserved for future release.

**Why this two-source approach?**

The combination of real-world queries (60) and PR-synthesized queries (18) reflects a deliberate trade-off between authenticity and coverage. The real-world queries guarantee ecological validity—they capture the exact patterns of challenge, ambiguity, and context that arise in professional practice. However, they are inherently limited by the specific experiences, domains, and task types encountered by the particular developers and researchers who contributed them. The PR-synthesized queries address this limitation by systematically sampling from a broader distribution of software development activities across 100 diverse repositories, ensuring that the training data covers development patterns that might not appear in the real-world query collection.

The 60/18 split (roughly 3:1 in favor of real-world queries) reflects a prioritization of authenticity over breadth. The authors trust that the 60 real-world queries capture core agentic patterns, and use the 18 synthesized queries primarily to fill coverage gaps and validate that the approach generalizes across query sources. The paper does not report an ablation that isolates the contribution of real-world versus synthesized queries, so the relative importance of each source remains an open question.

At the conclusion of this stage, the assembled query pool is:

$$Q = \{q_1, q_2, \ldots, q_{78}\}$$

where each \(q_i\) represents a multi-step collaborative task in either vibe coding or research workflow domains. The distribution across domains is visualized in Figure 4, though the paper does not specify the exact split. The query pool is now ready for trajectory collection.

---

#### Trajectory Collection Protocol

Given the assembled query pool, the next stage generates training trajectories—complete interaction sequences showing how an expert agentic system (GPT-5) collaborates with a human partner (a PhD annotator) to successfully resolve each query. This is where the "strategic curation" of the LIMI approach is most directly realized: the quality of the collected trajectories determines what the fine-tuned model ultimately learns about agentic behavior.

**Why SII CLI as the Execution Environment?**

The paper selects SII CLI (Lin et al., 2025) as the execution environment over alternatives like Claude Code and Gemini CLI based on four specific advantages:

1. **Comprehensive tool integration**: SII CLI supports both vibe coding and research workflows, providing a unified interface for the diverse tool ecosystem that real-world collaborative tasks require. For vibe coding tasks, this includes code editors, compilers, version control interfaces, and debugging tools; for research tasks, it includes data analysis libraries, visualization tools, literature search capabilities, and dataset management interfaces.

2. **Detailed trajectory logging capabilities**: The environment captures complete, fine-grained records of all interactions, which is "essential for high-quality training data collection" (Section 3.3). This means every reasoning output, every tool invocation (with its full parameters), every environmental observation (with its complete output), and every human feedback message is timestamped and preserved in the trajectory log.

3. **Flexible human-AI collaboration interfaces**: SII CLI supports "natural interaction patterns" (Section 3.3) between the human collaborator and the agentic model. This is important because the trajectory collection protocol relies on the annotator providing guidance, feedback, and clarification throughout the problem-solving process—mimicking the kind of collaborative interaction that occurs in real professional settings.

4. **Robust support for complex multi-step tasks**: The environment is designed for tasks requiring "coordinated tool usage" (Section 3.3) across extended interaction sequences, with state management that persists across tool invocations and session boundaries.

The SII CLI provides a "comprehensive toolkit that enables seamless collaboration across diverse knowledge work scenarios, integrating essential tools for software development, research activities, and information processing within a unified interface" (Section 3.3). This tool richness ensures that collected trajectories capture realistic collaborative contexts where effective task execution requires the agent to switch between different capabilities—writing code, running tests, analyzing data, searching for information, modifying plans based on results—rather than operating in an artificially constrained environment that simplifies away the complexity of real work.

**The Controlled Trajectory Collection Protocol**

The trajectory collection protocol is systematic and disciplined, designed to ensure that every collected trajectory represents successful task completion through authentic collaborative interaction. The protocol proceeds as follows:

For each query \(q_i\) in the training set:

1. **Setup**: The query is loaded into the SII CLI environment. The PhD student annotator assumes the role of the human collaborator—the domain expert who understands the task requirements, constraints, and success criteria. GPT-5 is configured as the agentic model within the environment, with access to all available tools.

2. **Iterative Collection**: The annotator and GPT-5 engage in collaborative problem-solving. The interaction follows a natural pattern: GPT-5 reasons about the current state of the task (model reasoning \(m_{i,j}\)), decides on a course of action, invokes appropriate tools (model tool calling \(t_{i,j}\)), receives results from the environment including any human feedback (environment observation \(o_{i,j}\)), then reasons about what to do next based on these observations. This cycle—reason, act, observe, reason—continues through the task.

3. **Persistence Until Success**: The protocol mandates that collection continues "until successful completion is achieved" (Section 3.3). This is a critical design choice. If the agent makes errors, invokes the wrong tools, produces incorrect outputs, or gets stuck, the annotator provides guidance—but the trajectory is not discarded. Instead, the full sequence, including the errors and the recovery process, is preserved. The paper argues that this persistence methodology "ensures that collected trajectories capture authentic human-AI interaction patterns, including natural back-and-forth communication, iterative refinement processes, and collaborative problem-solving strategies that characterize effective agentic behavior" (Section 3.3).

4. **Complete Sequence Recording**: For each query, the resulting trajectory \(\tau_i = \{a_{i,1}, \ldots, a_{i,n_i}\}\) captures the full interaction from initial query to successful resolution. The length \(n_i\) varies by query complexity: the average trajectory length is 42,400 tokens, with the longest reaching 152,000 tokens and the shortest at 13,000 tokens (Figure 4). These lengths reflect the substantial cognitive effort and extended interaction that real agentic tasks require—a single trajectory of 152,000 tokens contains hundreds of individual reasoning cycles, tool invocations, and environmental feedback events.

**Why persistence until success matters**

The choice to collect trajectories until successful completion, rather than capping interaction length or discarding failed attempts, is methodologically significant for three reasons:

- **It captures error recovery patterns.** Agentic intelligence is not just about executing correct plans—it is about recognizing when plans have gone wrong, diagnosing the source of failure, and adapting strategies accordingly. By preserving the full sequence including errors, the trajectories teach the model how to recover from specific types of failures that occur in real collaborative work: compilation errors in code, incorrect search results, model fitting failures, misinterpretation of task requirements.

- **It avoids survivorship bias.** If trajectories were collected only for tasks that the agent could solve trivially, the training data would be biased toward easy problems where the agent's initial plan is correct. The harder problems—the ones that require genuine agentic intelligence to navigate—would be underrepresented. The persistence protocol ensures that the training data contains examples of challenging, multi-step problem-solving where the path to success is non-obvious and requires adaptive strategy.

- **It provides positive-only demonstrations.** Every trajectory in the LIMI training set ends in success. This is a deliberate choice: the model is trained exclusively on examples of what good agentic behavior looks like—including the error recovery that was necessary along the way—rather than being exposed to failed trajectories that might teach undesirable patterns. The paper does not experiment with training on mixed positive/negative trajectories, so the importance of this choice remains empirically untested.

**The Role of the Human Collaborator**

The four PhD student annotators serving as human collaborators are not passive observers—they actively contribute to the trajectory through guidance, feedback, and clarification. This mimics the collaborative nature of real professional work, where a human might redirect an AI agent that is heading down an unproductive path, provide domain knowledge that the agent lacks, or clarify ambiguous requirements. However, the paper does not specify guidelines for how much guidance annotators should provide. This is a potential source of variability: trajectories collected by different annotators might differ in the amount and type of human intervention, which could affect what the model learns about autonomy versus reliance on human feedback.

The choice of GPT-5 as the agentic model during trajectory collection—rather than using the model that will be fine-tuned (GLM-4.5)—is an important design decision that the paper discusses only implicitly. GPT-5 is presumably a significantly more capable agent than GLM-4.5, which means the trajectories represent "expert demonstrations"—examples of what a highly capable agent would do in each situation, generated by a stronger model than the one being trained. This is a form of distillation: the fine-tuned model (LIMI) learns to approximate the behavior of a more capable system (GPT-5) by imitating its trajectories. The paper does not discuss whether collecting trajectories with GLM-4.5 itself (on-policy data) would yield different results, which is a notable methodological gap.

---

#### Training Procedure and Model Variants

With the full dataset \(\{(q_1, \tau_1), (q_2, \tau_2), \ldots, (q_{78}, \tau_{78})\}\) assembled, the LIMI approach proceeds to supervised fine-tuning. This is standard instruction tuning applied to agentic trajectories: the model is trained to predict the next token in each trajectory sequence, with the loss computed only on the model's output tokens (reasoning and tool calls), not on the environment observations or human feedback.

**Training Framework: Slime**

All fine-tuning experiments use the Slime framework (THUDM), which "provides robust and efficient infrastructure for supervised fine-tuning of large language models" and "ensures consistent training conditions, hyperparameter optimization, and convergence criteria across all experimental variants, enabling fair and reproducible comparison between different training approaches" (Section 4.1). The paper does not specify the exact hyperparameters used for fine-tuning (learning rate, batch size, number of epochs, optimizer settings), which is a significant omission. Without these details, reproducibility is compromised, and it is impossible to assess whether the observed performance differences might be attributable to differences in training configuration rather than data quality.

**Base Models for Fine-Tuning**

The primary base model is GLM-4.5, a 355B-parameter model described in Section 5.1 as providing "a unified approach to reasoning, coding, and agentic tasks, featuring hybrid reasoning modes and achieving 90.6% tool-calling success rate." When fine-tuned on the 78 LIMI trajectories, this becomes the model referred to simply as "LIMI."

To test generalization across model scales, the paper also fine-tunes GLM-4.5-Air, a 106B-parameter variant. The resulting model is referred to as "LIMI-Air." This enables a cross-scale comparison: if the LIMI data improves both the 355B and 106B models to similar relative degrees, it suggests that the strategic curation methodology captures fundamental agentic patterns that transfer regardless of model capacity.

**Comparative Datasets for Data Efficiency Experiments**

To validate the core claim that 78 curated samples outperform much larger datasets, the paper conducts comparative experiments by fine-tuning GLM-4.5 on three alternative datasets:

- **CC-Bench-trajectories** (Zeng et al., 2025): 260 samples. This dataset is associated with the CC-Bench benchmark and represents a modestly-sized alternative agentic training set.

- **AFM-WebAgent-SFT-Dataset** (PersonalAILab, 2024): 7,610 samples. This dataset targets web-based agentic interactions.

- **AFM-CodeAgent-SFT-Dataset** (PersonalAILab, 2024): 10,000 samples. This is the dataset that provides the 128× data ratio against LIMI's 78 samples—hence the paper's claim of "achieving superior agentic intelligence with 128 times fewer samples" (Section 1).

For clarity, models trained on these alternative datasets are referred to with suffixes: GLM-4.5-CC, GLM-4.5-Web, and GLM-4.5-Code. All are fine-tuned using the identical Slime training configuration to ensure that performance differences reflect data quality rather than implementation variations.

**What the model learns from the trajectories**

The training objective is standard causal language modeling (next-token prediction) on the concatenated query-trajectory sequences. There is no specialized agentic objective function, reinforcement learning component, or auxiliary loss. The model learns agentic behavior purely through imitation: by observing the patterns of reasoning, tool invocation, and response to environmental feedback in the expert demonstrations, it internalizes the decision-making process that leads to successful task completion.

This imitation-based approach has both strengths and limitations that the paper does not discuss explicitly:

- **Strength**: It is simple, reproducible, and leverages the same training infrastructure used for standard fine-tuning. There are no complex reward design, exploration strategies, or credit assignment problems to solve.

- **Limitation**: The model can only imitate what it sees in the training data. If the training trajectories exhibit specific patterns—certain types of error recovery strategies, particular ways of interacting with tools, specific communication styles with the human collaborator—the fine-tuned model will reproduce these patterns. Whether it can generalize to genuinely novel task types, tool interfaces, or error conditions not represented in the 78 trajectories is an empirical question that the paper addresses only partially through the generalization benchmarks.

---

#### Evaluation Framework

The paper evaluates LIMI across two complementary assessment strategies to comprehensively validate both the core claim (strategic curation yields superior agentic intelligence) and the robustness of the approach (whether improvements are limited to the training domains or generalize broadly).

**Primary Evaluation: AgencyBench**

AgencyBench (Li et al., 2025b) is "specifically designed for assessing agentic capabilities in collaborative scenarios" and contains "carefully curated tasks that reflect the complexity and collaborative nature of real-world agentic scenarios across both vibe coding and research workflows" (Section 3.4). The benchmark comprises 10 tasks (Table 1), with four in the vibe coding category and six in the research workflow category. Each task contains multiple subtasks that must be completed sequentially, with each subtask building on the previous ones.

The 10 tasks are:

**Vibe Coding (Tasks 1–4):**
1. Build a C++ console chat system with escalating features (user registration, friend management, chat history, global search, concurrent messaging)
2. Create a Java console task management application (user system, basic CRUD, advanced filtering/sorting, full-text search, multi-user concurrency)
3. Develop a web-based Gomoku game (board rendering, win detection, undo/replay, basic AI, advanced AI with Minimax and iterative deepening)
4. Build a local microservice pipeline (deterministic event generator, transactional KV store, orchestrator/worker system, symbolic planner, autonomous fault detection and self-repair)

**Research Workflows (Tasks 5–10):**
5. Compare LLM performance on the DynToM dataset (download data, write API calls, select samples, test multiple models, compute accuracy)
6. Conduct a comprehensive comparative study of standard versus reasoning-enabled LLMs (statistical analysis, RMS metrics, ECE calibration, NLL, comprehensive report)
7. Search Hugging Face for datasets matching specific criteria and extract metadata (three subtasks with different search criteria: philosophy QA, academic multiple-choice, hate speech detection)
8. Discover mathematical function structures for scientific data (iteratively modify equations until loss falls below progressively tighter thresholds: \(10^{-3}\), \(10^{-5}\), \(10^{-6}\), \(10^{-7}\))
9. Answer complex multi-conditional questions about NBA players using web search and reasoning (four subtasks with different player identification criteria)
10. Answer in-depth business questions about S&P 500 companies using financial data and leadership information (four subtasks with different company identification criteria)

Each task is evaluated using three metrics that collectively capture both effectiveness and efficiency dimensions of collaborative intelligence:

**First-Turn Functional Completeness (FTFC):**
This metric measures "the percentage of requirements correctly implemented in the initial response" (Section 3.4). It assesses whether the model can produce a substantially correct solution on its first attempt, without requiring iterative refinement through multiple rounds of interaction. A high FTFC indicates that the model quickly grasps the full scope of the task and produces a comprehensive initial solution—an important capability for deployment scenarios where interaction rounds are expensive or where the human collaborator expects rapid progress.

**Success Rate at R Rounds (SR@R):**
This metric represents "the percentage of queries successfully completed within R allocated rounds" (Section 3.4). The paper sets \(R = 3\) "to balance iterative refinement with computational efficiency" (Section 3.4). A round in this context is one complete cycle of model reasoning → tool invocation → environmental observation → human feedback. SR@3 therefore measures whether the model can complete the task within three such cycles. Tasks that require more than three rounds are counted as failures. This metric captures the model's efficiency in navigating the task—a model that eventually succeeds but requires many rounds of back-and-forth would score poorly on SR@3 even if the ultimate output is correct.

**Remaining Chances at R Rounds (RC@R):**
This metric calculates "the average number of unused rounds when queries are successfully completed" (Section 3.4). It is computed only for successfully completed queries (those that finish within R rounds). If a query is successfully completed in 1 round, the remaining chances are \(R - 1 = 2\); if completed in 3 rounds, the remaining chances are 0. A higher RC@R indicates that the model is not only successful but efficient—it completes tasks using fewer interaction rounds than the budget allows, suggesting that it plans effectively, makes good initial decisions, and requires minimal correction. The metric thus captures "computational efficiency" in the interaction dimension: models that are faster and more autonomous score higher on RC@R.

The combination of these three metrics provides a nuanced picture of agentic capability. A model might have high FTFC but low SR@3 if its initial solutions are often correct in structure but fail on edge cases that require refinement. Conversely, a model might have low FTFC but high SR@3 if it rarely gets things right on the first try but reliably converges to correct solutions within three rounds of iterative refinement. The "AVG" column in Table 2 appears to be a simple average of the three metrics (FTFC, SR@3, and RC@3), though the paper does not explicitly define the aggregation.

**Generalization Assessment: Established Benchmarks**

To test whether LIMI's improvements are specific to the AgencyBench tasks or reflect broader capability gains, the paper evaluates on six established benchmarks spanning diverse agentic and coding scenarios:

- **tau2-bench-airline** and **tau2-bench-retail** (Yao et al., 2024a; Barres et al., 2025): Benchmarks for conversational tool-use agents, testing the model's ability to interact with simulated airline and retail systems through natural language dialogue. The metric is Pass@4 accuracy, "defined as the fraction of 4 independent runs that succeed" (Section 3.4), which measures consistency rather than single-sample performance.

- **EvalPlus-HumanEval** and **EvalPlus-MBPP** (Liu et al., 2024, 2023): Code generation benchmarks that test the model's ability to produce correct Python functions from natural language specifications. The metric is standard accuracy (pass rate on test cases).

- **DS-1000** (Lai et al., 2022): A benchmark for data science code generation, testing the model's ability to generate correct solutions for data manipulation, analysis, and visualization tasks using Python libraries like pandas, numpy, and matplotlib. The metric is accuracy.

- **SciCode** (Tian et al., 2024): A scientific computing benchmark that tests code generation for research-oriented tasks. The paper reports two sub-metrics: SciCode-MP (Main Problem) and SciCode-SP (Sub Problem). The metric is accuracy.

These benchmarks cover a range of capabilities that are components of agentic intelligence—tool use, code generation, data science, scientific computing—but do not themselves require the integrated, multi-turn, collaborative problem-solving that AgencyBench tests. Strong performance on these benchmarks would indicate that LIMI's training on collaborative trajectories produces transferable improvements in the underlying skills (reasoning, planning, code generation, tool usage) rather than merely teaching the model to navigate the specific interaction patterns of the SII CLI environment.

**Evaluation Configurations: With and Without CLI Access**

A critical aspect of the evaluation design is the comparison between two operational conditions:

- **With SII CLI environment access**: The model has access to the full tool ecosystem provided by SII CLI, including code execution, file management, data analysis libraries, and search capabilities. This configuration is used for AgencyBench evaluation (since AgencyBench tasks inherently require tool interaction) and for the primary generalization benchmark results reported in Table 3.

- **Without SII CLI environment access**: The model operates in a standard inference setting without tool augmentation, relying solely on its intrinsic capabilities. This configuration is used for the generalization benchmarks (Section 4.5, Table 4) to "isolate the contribution of tool-enhanced capabilities versus intrinsic model reasoning abilities" (Section 4.1).

This dual-environment evaluation serves as an important ablation. If LIMI's improvements were entirely dependent on the SII CLI environment—if the fine-tuning simply taught the model how to use specific tools—then performance without CLI access would be similar to the base model. Conversely, if LIMI's improvements reflect genuine enhancements in reasoning, planning, and problem decomposition, then benefits should persist even without tool access, albeit at a lower absolute level.

The results bear out the latter interpretation (Section 4.5): LIMI maintains competitive advantage even without CLI access, achieving 50.0% average performance compared to GLM-4.5's 48.7% on the generalization benchmarks. The 7.2 percentage point improvement when CLI access is available (57.2% vs. 50.0%) represents the model's "enhanced ability to leverage environmental resources effectively" (Section 4.5)—the tool-use amplification of capabilities that already exist. This dual benefit—intrinsic improvement plus tool-use skill—is what the paper claims as evidence for the synergistic nature of its approach.

**Evaluation on AgencyBench for Baseline Models**

The baseline models (Kimi-K2-Instruct, DeepSeek-V3.1, Qwen3-235B-A22B-Instruct, GLM-4.5) are evaluated on AgencyBench in their off-the-shelf state, without fine-tuning on the LIMI dataset. The paper does not specify whether these models were given SII CLI access during evaluation, but the context implies they were—the AgencyBench tasks require tool interaction, and without CLI access, completion would be impossible. The baseline models' substantially lower performance (GLM-4.5 at 45.1%, others ranging from 11.9% to 27.5%) establishes that even strong pretrained models with tool access struggle on the integrated, multi-turn collaborative tasks that AgencyBench demands, and that LIMI's fine-tuning on curated agentic trajectories provides capabilities that pretraining alone does not confer.

---

#### Summary of Design Choices and Their Justifications

- **78 training samples, not more**: The paper's core hypothesis is that strategic curation beats scale. Using 78 samples rather than, say, 500 or 1,000 is designed to make the efficiency claim as stark as possible—the dramatic contrast with 10,000-sample training loses force if LIMI used a more moderate number. The risk is that 78 might be insufficient for robustness; the paper mitigates this by ensuring each trajectory is long (average 42.4K tokens) and rich with decision points.

- **Human-AI collaborative trajectories rather than solo agent traces**: The collaborative protocol captures patterns of human guidance, feedback interpretation, and adaptive strategy that solo trajectories would miss. This matters because the fine-tuned model will operate in collaborative settings and needs to know how to process and respond to human input, not just execute autonomously.

- **GPT-5 as trajectory generator, GLM-4.5 as training target**: This distillation approach leverages a stronger model's capabilities to generate expert demonstrations that a weaker model can learn from. The alternative—collecting trajectories with GLM-4.5 itself—would produce lower-quality demonstrations (since GLM-4.5 is a weaker agent) and might fail to complete the harder tasks even with human guidance.

- **Persistence until success in trajectory collection**: By ensuring every training trajectory ends in successful task completion, the dataset provides exclusively positive examples of agentic behavior. The model never sees examples of giving up, making irrecoverable errors, or failing to complete tasks—it only sees patterns that ultimately lead to success, including the error recovery that was necessary along the way.

- **SII CLI as execution environment over alternatives**: The choice prioritizes comprehensive tool integration and detailed logging. Claude Code or Gemini CLI might have different tool ecosystems, which could affect what agentic patterns the trajectories capture and whether the fine-tuned model's tool-use skills transfer to other environments.

- **AgencyBench as primary evaluation over existing benchmarks**: AgencyBench is specifically designed for the collaborative, multi-turn, integrated agentic tasks that the paper targets. Existing benchmarks (tau2-bench, HumanEval, DS-1000, SciCode) test individual capability dimensions in isolation, which would miss the holistic agentic intelligence that LIMI is designed to cultivate.

- **Dual-environment evaluation (with/without CLI)**: This ablation separates intrinsic capability improvement from tool-use amplification, providing evidence that LIMI's training improves foundational reasoning rather than merely teaching environment-specific tool patterns.

- **Slime framework for all fine-tuning**: Using a consistent training infrastructure across all comparative experiments ensures that performance differences reflect data quality, not variations in optimization hyperparameters, hardware configuration, or implementation details.

## 4. Key Insights and Innovations

### Innovation 1: The Agency Efficiency Principle — Agentic Intelligence Does Not Follow Data Scaling Laws

The paper's most fundamental contribution is not a new training algorithm or model architecture but an **empirically grounded refutation of the default paradigm** in agentic AI development. The dominant assumption—inherited directly from language modeling scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) and reinforced by the data-intensive training pipelines of models like GLM-4.5, Kimi-K2, and the AFM family (Section 5.1)—has been that more interaction trajectories, more tool-use demonstrations, and more collaborative scenarios produce better autonomous agents. This assumption has driven the field toward increasingly large-scale data synthesis efforts, with models trained on thousands to tens of thousands of agentic trajectories under the implicit belief that agency emerges as a function of exposure volume.

LIMI demonstrates that this assumption is not merely imprecise but **qualitatively wrong** for agentic intelligence. Using only 78 carefully curated training samples, LIMI achieves 73.5% on AgencyBench—a benchmark explicitly designed to test integrated, multi-turn collaborative agentic capabilities—while a model trained on 10,000 samples from the AFM-CodeAgent-SFT-Dataset achieves only 47.8% (Table 2). This is not a marginal efficiency gain; it is a 53.7 percentage point improvement achieved with **128× fewer training examples**. The direction of the relationship between data quantity and agentic performance is effectively inverted relative to conventional wisdom: the model trained on less data performs dramatically better.

What makes this more than a data efficiency finding is the formulation of the **Agency Efficiency Principle**: "machine autonomy emerges not from data abundance but from strategic curation of high-quality agentic demonstrations." This is a conceptual claim about the nature of agency itself. It posits that agentic intelligence is not an emergent property that appears at sufficient scale—like the phase transitions observed in language model pretraining—but rather a capability that must be **elicited through demonstrations that capture its essential structure**. The implication is that understanding the essence of agency matters more than accumulating training examples, a claim with significant consequences for research prioritization.

The finding draws a direct lineage to LIMA (Zhou et al., 2023) and LIMO (Ye et al., 2025), but extends the Less-Is-More paradigm into a domain where the capabilities in question are substantially more complex than single-turn language generation or mathematical reasoning. LIMA showed 1,000 examples suffice for alignment; LIMO showed 817 examples suffice for mathematical reasoning; LIMI shows 78 examples suffice for agency—which requires planning, tool orchestration, multi-step execution, environmental feedback processing, and collaborative communication **within a single integrated workflow**. That the sample count decreases as capability complexity increases is not coincidental: it suggests that the more structured and goal-directed the target behavior, the more demonstration quality dominates over quantity, because the space of possible trajectories that exhibit genuine agency is far more constrained than the space of possible language outputs or mathematical reasoning chains.

The significance extends beyond the specific empirical result because it identifies **what constitutes a high-quality agentic demonstration**. The paper's data construction methodology—real-world queries from professional practice, persistent-till-success trajectory collection, human-AI collaborative interaction patterns—operationalizes a theory of what agency looks like in practice. This is a diagnostic contribution: the field now has a concrete template for what strategic curation means in the agentic context, which can guide future work even if the exact query count varies by domain.

Evidence anchoring: Table 2 (AgencyBench results) shows LIMI at 73.5% vs. GLM-4.5-Code at 47.8% (78 vs. 10,000 samples). Table 3 (generalization benchmarks) shows this advantage generalizes: LIMI at 57.2% average vs. GLM-4.5-Code at 40.9%. The consistent pattern across both in-domain (AgencyBench) and out-of-domain (tau2-bench, HumanEval, DS-1000, SciCode) evaluations rules out the possibility that the 78 samples simply overfit to AgencyBench's specific task distributions.

---

### Innovation 2: Reconceptualizing Agentic Data Collection as Trajectory Curation Rather Than Instance Accumulation

The second conceptual contribution is a reframing of what constitutes a "training example" for agentic intelligence—and, by extension, what it means to build a high-quality agentic training dataset. Prior work in agentic AI (including the AFM family, CC-Bench, and the training pipelines underlying GLM-4.5 and Kimi-K2) has treated agentic training data as collections of interaction instances: individual tool calls, isolated code completions, single-turn task executions. The implicit model has been that exposing a model to many such instances—each demonstrating some fragment of agentic behavior—will cause the full capability to emerge through aggregation.

LIMI's trajectory collection protocol (Section 3.3) embodies a fundamentally different model. Each of the 78 training examples is not an isolated interaction fragment but a **complete, end-to-end trajectory of successful collaborative problem-solving**, with an average length of 42,400 tokens and a maximum of 152,000 tokens (Figure 4). These trajectories capture the full dynamic of agentic work: initial task understanding, planning and decomposition, iterative tool invocation, interpretation of environmental feedback, error detection and recovery, strategy adaptation, and collaborative communication with the human partner. Each trajectory is a coherent narrative of problem-solving rather than a snapshot of a single action.

This reframing has profound implications for how the field thinks about data quality in agentic AI:

**Density of learning signal.** A single 42,400-token trajectory contains hundreds of individual decision points—each reasoning step, each tool invocation, each response to environmental feedback—that collectively demonstrate the full pattern of agentic behavior. The paper's longest trajectory (152,000 tokens) encodes more agentic decision points than hundreds or thousands of isolated interaction fragments. The "78 samples" figure is therefore misleading in a productive way: the quantity is small by instance count but enormous by decision-point count. The field's conventional unit of "a training sample" (a single query-response pair or tool invocation) is the wrong unit for measuring agentic training data, and the paper's results demonstrate this by showing that a dataset measured in traditional samples outperforms a dataset measured in trajectories.

**Temporal coherence as a quality signal.** By collecting trajectories that persist until successful completion, the paper ensures that each training example demonstrates the **full problem-solving arc**—including errors, recovery, and adaptation—rather than just successful actions. A trajectory that includes the model making a wrong tool call, receiving an error, diagnosing the problem, and trying a different approach teaches something that 100 isolated successful tool calls cannot: how to respond when things go wrong. The paper's "persistence until success" protocol is not just a collection methodology but a theory of what makes a trajectory valuable for learning agency—it must include the signature patterns of genuine agentic intelligence, which include error recovery as much as correct execution.

**The human collaborator as trajectory co-author.** The collaborative protocol—where PhD annotators work alongside GPT-5 in the SII CLI environment—produces trajectories that encode human judgment about when to intervene, what feedback to provide, and how to redirect unproductive paths. These patterns are themselves part of what the model learns: not just how to execute tasks autonomously, but how to **collaborate effectively with a human partner**. This distinguishes LIMI's training data from purely synthetic agentic trajectories generated without human involvement, and from reinforcement learning approaches where the agent explores independently. The model learns from demonstrations of good human-AI collaboration, not just good AI solo performance.

Evidence anchoring: Figure 4 shows the trajectory length distribution (min 13K, max 152K, average 42.4K tokens), establishing that the 78 samples represent substantially more learning signal than the sample count alone suggests. The contrast with the AFM datasets (10,000 and 7,610 samples respectively) in Table 2 demonstrates that trajectory quality dominates instance count, even when the instance-count ratio is 128:1.

---

### Innovation 3: The Synergy Principle — Agentic Training Simultaneously Improves Intrinsic Capability and Tool-Use Proficiency

The third conceptual contribution emerges from the dual-environment evaluation in Section 4.5 and represents a finding about **what agentic training actually teaches the model**. The paper shows that LIMI's benefits are not confined to the trained environment (SII CLI) but generalize to standard inference settings without tool access, and that the magnitude of improvement is **additive rather than substitutive**: the model gains both better foundational reasoning (visible without tools) and better ability to leverage tools (visible with tools), and these gains compound rather than trading off against each other.

Specifically, LIMI achieves 50.0% average performance without CLI access versus 57.2% with CLI access (Tables 4 and 3 respectively), representing a 7.2 percentage point tool-use amplification on top of the intrinsic improvement. The intrinsic improvement itself is substantial: LIMI outperforms all external baseline models in the tool-free setting (50.0% vs. GLM-4.5 at 48.7%, Kimi-K2-Instruct at 40.3%, DeepSeek-V3.1 at 36.5%, Qwen3-235B-A22B-Instruct at 37.3%). This dual improvement pattern has important implications:

**It resolves a potential confound in the main result.** A skeptical interpretation of LIMI's AgencyBench performance might be that the fine-tuning simply taught the model to use SII CLI's specific tool interfaces, and that the performance advantage reflects environment-specific optimization rather than genuine agentic capability. The tool-free evaluation (Table 4) largely refutes this: if LIMI's advantage were purely tool-specific, it should disappear or reverse without tool access. Instead, LIMI maintains superiority, establishing that the improvement is at the level of reasoning, planning, and problem decomposition—the cognitive substrate of agency, not just the interface layer.

**It challenges the implicit separation between "reasoning" and "acting" in agentic training.** Much of the agentic AI literature treats these as distinct capabilities trained through separate procedures: reasoning through pretraining and instruction tuning, acting through tool-use fine-tuning or reinforcement learning. LIMI's training on complete collaborative trajectories—where reasoning, acting, and environmental feedback are interleaved—produces simultaneous improvement in both. This suggests that the separation may be artificial, and that the most efficient path to agentic intelligence integrates these capabilities in training as they are integrated in execution.

**It suggests a mechanism for the data efficiency finding.** If agentic training improves both intrinsic reasoning and tool-use skill simultaneously, a single high-quality trajectory provides learning signal on multiple capability dimensions concurrently. A trajectory that demonstrates how to reason about a task, select appropriate tools, interpret results, and adapt strategy provides more learning per token than training that targets each dimension separately. This may partially explain why 78 rich trajectories outperform thousands of narrower training instances: the dense, multi-dimensional signal in each trajectory compounds across the training set.

Evidence anchoring: Table 4 vs. Table 3 shows the 7.2 percentage point gap between tool-free and tool-augmented performance. The consistent pattern across all generalization benchmarks (not just AgencyBench) establishes that the intrinsic improvement is broad rather than task-specific.

---

### Innovation 4: GitHub Pull Requests as a Scalable, Authentic Source of Agentic Queries

While not the headline contribution, the PR-based query synthesis methodology (Section 3.2, with the full prompt in Appendix B) represents a **methodological innovation with independent value** for the agentic AI research community. The problem of generating realistic, diverse, and verifiably authentic agentic queries is a bottleneck for data-efficient approaches like LIMI: if the approach relies on strategic curation, the field needs systematic methods for producing high-quality queries beyond ad-hoc collection from practitioners.

The innovation is not the use of GitHub PRs per se—the field has used code repositories for training data before—but rather the **transformation of retrospective code changes into prospective collaborative tasks** through a structured synthesis pipeline with systematic quality controls. The synthesis prompt (Appendix B) provides a reusable template for converting any PR into a forward-looking agentic query that captures the technical requirements, domain context, and success criteria of the original development task. The quality assurance protocol—four PhD annotators evaluating semantic alignment between generated queries and PR content, followed by strategic sampling for domain coverage—provides a template for how to filter synthesized queries to maintain authenticity while expanding coverage.

The significance of this contribution extends beyond LIMI itself. The paper notes that "several thousand high-quality synthetic queries" were generated through the pipeline, of which only 18 were used for LIMI training. The remaining queries and their trajectory data "will be released in future work to benefit the broader research community" (Section 3.2). This positions the PR synthesis methodology as a **community resource** rather than a paper-specific tool, potentially enabling other researchers to build agentic training datasets without the cost and difficulty of collecting real-world queries from professional practitioners.

The pipeline's design choices reflect careful reasoning about what makes a query "agentic." The repository selection threshold (10,000+ stars) ensures codebase quality; the domain diversification across frontend, backend, deployment, debugging, and optimization ensures that queries span the range of professional development activities; the complexity filter (below 1,200 token diffs, excluding Markdown-only changes) ensures queries represent focused, substantive tasks rather than trivial or purely documentation changes. These filters operationalize a theory of what constitutes a worthwhile agentic query: one that requires genuine technical intervention in a production-quality codebase, matching the scope of what a human developer would realistically delegate to an AI collaborator.

Evidence anchoring: Section 3.2 describes the full pipeline with specific thresholds (10,000+ stars, 1,200 token cutoff, 100 repositories, 1,000 PRs per repository, 100 sampled per repository for synthesis). Appendix B provides the complete synthesis prompt, making the methodology fully reproducible. The claim that "several thousand high-quality synthetic queries" were generated is not independently verified in the paper but is consistent with the pipeline scale.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluation dataset is AgencyBench (Li et al., 2025b), a benchmark comprising 10 multi-step collaborative tasks (4 vibe coding, 6 research workflow) designed to test integrated agentic capabilities including planning, tool orchestration, multi-turn interaction, and adaptive problem-solving. Each task contains multiple subtasks that build on each other sequentially, requiring sustained autonomous execution across extended interaction sequences. For generalization assessment, the paper evaluates on six established benchmarks: tau2-bench-airline and tau2-bench-retail (Yao et al., 2024a; Barres et al., 2025) for conversational tool use, EvalPlus-HumanEval and EvalPlus-MBPP (Liu et al., 2024, 2023) for code generation, DS-1000 (Lai et al., 2022) for data science code generation, and SciCode (Tian et al., 2024) for scientific computing.

- **Base model(s).** The primary base model is GLM-4.5, a 355B-parameter model that provides "a unified approach to reasoning, coding, and agentic tasks, featuring hybrid reasoning modes and achieving 90.6% tool-calling success rate" (Section 5.1). For cross-scale generalization analysis, the paper also fine-tunes GLM-4.5-Air, a 106B-parameter variant. The authors select GLM-4.5 because it represents a strong contemporary foundation model with non-trivial baseline agentic performance (45.1% on AgencyBench in its off-the-shelf state), leaving substantial room for improvement through fine-tuning while not being so capable that gains would be marginal. Baseline models for comparison include Kimi-K2-Instruct (1T parameters, MoE architecture), DeepSeek-V3.1 (671B), Qwen3-235B-A22B-Instruct (235B, MoE), and the base GLM-4.5.

- **Metrics.** AgencyBench evaluation uses three complementary metrics (Section 3.4): First-Turn Functional Completeness (FTFC) measures the percentage of requirements correctly implemented in the initial response, capturing whether the model produces a substantially correct solution without iterative refinement; Success Rate at R rounds (SR@R, with R=3) represents the percentage of queries successfully completed within three interaction rounds, measuring efficiency in achieving task completion; and Remaining Chances at R rounds (RC@R) calculates the average number of unused rounds when queries are successfully completed, capturing computational efficiency in the interaction dimension. The paper reports an "AVG." column that appears to be an unweighted average of FTFC, SR@3, and RC@3, though the aggregation formula is not explicitly stated. For generalization benchmarks, standard metrics are used: Pass@4 accuracy on tau2-bench (fraction of 4 independent runs that succeed), accuracy on EvalPlus-HumanEval, EvalPlus-MBPP, and DS-1000, and accuracy on SciCode-MP (Main Problem) and SciCode-SP (Sub Problem) separately.

- **Baselines.** The paper evaluates against four off-the-shelf foundation models in their default instruction-tuned states without any additional fine-tuning: Kimi-K2-Instruct (Team et al., 2025), DeepSeek-V3.1 (Yang et al., 2025), Qwen3-235B-A22B-Instruct (Yang et al., 2025), and GLM-4.5 (Zeng et al., 2025). For data efficiency comparisons, additional baselines are created by fine-tuning GLM-4.5 on three alternative datasets using identical training configurations within the Slime framework: GLM-4.5-CC (trained on CC-Bench-trajectories with 260 samples), GLM-4.5-Web (trained on AFM-WebAgent-SFT-Dataset with 7,610 samples), and GLM-4.5-Code (trained on AFM-CodeAgent-SFT-Dataset with 10,000 samples). These serve as scale-based baselines to test the core claim that 78 curated samples outperform much larger datasets.

- **Generation budget / compute accounting.** The paper does not measure or report computational budgets in FLOPs, GPU-hours, or inference tokens. The primary axis of comparison is training dataset size (number of samples), not inference-time computation. All models are evaluated on the same benchmarks under identical conditions, so the comparison is on performance per unit of training data rather than performance per unit of training compute. This is a notable omission: the paper does not report training time, computational cost, or FLOPs consumed during fine-tuning for any model variant, making it impossible to assess whether the data efficiency translates to compute efficiency in training. The evaluation protocol uses fixed interaction round budgets (R=3 for AgencyBench SR@3 and RC@3) rather than variable compute allocation.

- **Cross-validation / statistical protocol.** The paper does not describe any cross-validation, statistical significance testing, confidence intervals, or multiple-run averaging for AgencyBench results. Performance is reported as single-point estimates without error bars or variance measures. The generalization benchmarks use Pass@4 for tau2-bench (averaging over 4 independent runs), providing some robustness for those metrics, but AgencyBench metrics appear to be based on single evaluation runs. The four PhD annotators provide quality assurance during data construction but not statistical validation of results. The absence of any statistical protocol is a significant methodological gap: with 78 training samples and 10 AgencyBench tasks, the performance estimates could have substantial variance, and the reported differences between models cannot be assessed for statistical reliability.

### Main Quantitative Results

#### AgencyBench Performance: LIMI vs. State-of-the-Art Models

The headline result is that LIMI achieves 73.5% average performance on AgencyBench, dramatically outperforming all baseline models. Table 2 reports the full breakdown:

- LIMI (355B, 78 samples): FTFC 71.7%, RC@3 74.2%, SR@3 74.6%, average 73.5%
- GLM-4.5 (355B, no fine-tuning): FTFC 37.8%, RC@3 50.0%, SR@3 47.4%, average 45.1%
- Qwen3-235B-A22B-Instruct (235B, no fine-tuning): FTFC 23.0%, RC@3 28.2%, SR@3 31.3%, average 27.5%
- Kimi-K2-Instruct (1T, no fine-tuning): FTFC 20.7%, RC@3 25.1%, SR@3 26.6%, average 24.1%
- DeepSeek-V3.1 (671B, no fine-tuning): FTFC 10.6%, RC@3 11.9%, SR@3 13.3%, average 11.9%

The 28.4 percentage point gap between LIMI and its base model GLM-4.5 (73.5% vs. 45.1%) represents the improvement attributable to fine-tuning on the 78 curated trajectories. The performance advantage is consistent across all three AgencyBench metrics, with the largest gap in FTFC (71.7% vs. 37.8%, a 33.9 percentage point improvement), indicating that LIMI's improvements are most pronounced in first-attempt completeness—the model's ability to produce substantially correct solutions without iterative refinement.

The gap between LIMI and the strongest external baseline (Kimi-K2-Instruct at 24.1%) is 49.4 percentage points, establishing that LIMI's capabilities on AgencyBench are not simply a function of model scale (Kimi-K2-Instruct is a 1T-parameter MoE model, substantially larger than the 355B GLM-4.5 that LIMI is based on) or pretraining alone. The poor performance of DeepSeek-V3.1 (11.9%) and Qwen3-235B-A22B-Instruct (27.5%) further reinforces that strong language modeling capability does not automatically translate to strong agentic performance on integrated collaborative tasks.

#### Generalization Performance Across Established Benchmarks

The generalization results in Table 3 demonstrate that LIMI's improvements extend beyond the AgencyBench tasks to diverse benchmarks spanning tool use, code generation, and scientific computing. The headline comparison: LIMI achieves 57.2% average performance across all benchmarks (including AgencyBench) versus GLM-4.5 at 43.0%.

The per-benchmark breakdown shows LIMI leading on most metrics:

- tau2-bench-airline: LIMI 34.0% vs. GLM-4.5 28.0% vs. Kimi-K2-Instruct 38.0%
- tau2-bench-retail: LIMI 45.6% vs. GLM-4.5 36.8% vs. Kimi-K2-Instruct 28.9%
- DS-1000: LIMI 36.6% vs. GLM-4.5 33.6% vs. DeepSeek-V3.1 42.4%
- EvalPlus-HumanEval: LIMI 92.1% vs. GLM-4.5 90.2% vs. Kimi-K2-Instruct 92.1%
- EvalPlus-MBPP: LIMI 82.3% vs. GLM-4.5 79.6% vs. Qwen3-235B-A22B-Instruct 81.7%
- SciCode-MP: LIMI 3.1% vs. GLM-4.5 1.5% vs. Kimi-K2-Instruct 3.1%
- SciCode-SP: LIMI 25.3% vs. GLM-4.5 25.3% vs. Kimi-K2-Instruct 23.6%

Several patterns merit attention. First, LIMI's advantages are largest on tool-use benchmarks (tau2-bench-retail: +8.8 percentage points over GLM-4.5) and relatively modest on pure code generation benchmarks (EvalPlus-HumanEval: +1.9 points, EvalPlus-MBPP: +2.7 points). This is consistent with the training data's focus on collaborative workflows involving tool orchestration, which would be expected to transfer most directly to tool-use tasks. The small gains on code generation suggest that the 78 trajectories may not substantially improve underlying programming ability beyond the base model's pretrained capabilities—the improvement is more in the orchestration of tools and the management of multi-step workflows than in the fundamental skill of writing code.

Second, Kimi-K2-Instruct outperforms LIMI on tau2-bench-airline (38.0% vs. 34.0%) and matches on EvalPlus-HumanEval (92.1% each), despite LIMI's overwhelming advantage on AgencyBench. This indicates that LIMI's strengths are domain-aligned with its training data (collaborative software development and research workflows) and do not uniformly dominate across all agentic capability dimensions. The paper does not discuss these comparative weaknesses, which merit acknowledgment.

Third, SciCode performance is extremely low across all models (LIMI at 3.1% on Main Problems, 25.3% on Sub Problems). The base GLM-4.5 and LIMI achieve identical SciCode-SP performance (25.3%), suggesting that the fine-tuning does not improve scientific computing code generation. This may reflect the domain gap between the training trajectories (which focus on software development and research workflows at a higher level of abstraction) and SciCode's requirements (scientific computing tasks curated by scientists, requiring domain-specific knowledge that the 78 trajectories likely do not cover).

#### Data Efficiency: 78 Samples vs. 10,000 Samples

The core data efficiency results in Table 2 provide the empirical foundation for the paper's central claim. Comparing LIMI (78 samples) against models fine-tuned on alternative datasets using identical training configurations:

- GLM-4.5-Code (10,000 samples): FTFC 48.0%, RC@3 48.0%, SR@3 47.5%, average 47.8%
- GLM-4.5-Web (7,610 samples): FTFC 36.7%, RC@3 36.7%, SR@3 36.7%, average 36.7%
- GLM-4.5-CC (260 samples): FTFC 30.4%, RC@3 30.4%, SR@3 26.7%, average 29.2%
- LIMI (78 samples): FTFC 71.7%, RC@3 74.2%, SR@3 74.6%, average 73.5%

The most dramatic comparison is LIMI vs. GLM-4.5-Code: a 53.7 percentage point improvement (73.5% vs. 47.8%) using 128× fewer samples. This is the basis for the paper's claim of "achieving superior agentic intelligence with 128 times fewer samples" (Section 1).

Notably, GLM-4.5-Web (7,610 samples) and GLM-4.5-CC (260 samples) both underperform their base model GLM-4.5 (45.1%), which received no fine-tuning. Models trained on these datasets achieve AgencyBench averages of 36.7% and 29.2% respectively, both below the 45.1% of the untuned GLM-4.5. This is a striking negative result: fine-tuning on thousands of agentic trajectories from existing datasets **degrades** AgencyBench performance relative to the base model. The paper does not discuss this finding explicitly, but it suggests that the alternative datasets may train the model on patterns of agentic behavior that are mismatched with AgencyBench's task structure—potentially teaching interaction patterns, tool-use conventions, or problem-solving strategies that conflict with what AgencyBench requires. This negative result, if robust, actually strengthens the paper's central argument: not only does strategic curation outperform scale, but indiscriminate scale can be actively harmful, reducing performance below the untuned baseline.

The generalization benchmark results (Table 3) mirror the data efficiency pattern, though with smaller absolute gaps:

- LIMI (78 samples): 57.2% average
- GLM-4.5-Code (10,000 samples): 40.9% average
- GLM-4.5-CC (260 samples): 39.2% average
- GLM-4.5-Web (7,610 samples): 33.7% average

LIMI's relative improvement over GLM-4.5-Code on generalization is 39.9% (57.2% vs. 40.9%), somewhat smaller than the 53.7% relative improvement on AgencyBench alone, suggesting that the data efficiency advantage is most pronounced in-domain (AgencyBench) and partially but not fully transfers to out-of-domain benchmarks. The per-benchmark breakdown reveals an interesting anomaly: GLM-4.5-CC achieves 38.0% on tau2-bench-airline, outperforming both LIMI (34.0%) and the base GLM-4.5 (28.0%), despite its poor overall average. This indicates that the CC-Bench trajectories dataset, while broadly less effective than LIMI's curated data, provides useful training signal for specific tool-use interaction patterns—a nuanced finding that the paper does not explore.

#### Generalization Across Model Scales

The cross-scale results in Table 2 demonstrate that the LIMI approach transfers to a smaller model variant:

- GLM-4.5-Air (106B, no fine-tuning): FTFC 15.0%, RC@3 16.1%, SR@3 20.0%, average 17.0%
- LIMI-Air (106B, 78 samples): FTFC 35.4%, RC@3 34.3%, SR@3 33.1%, average 34.3%

The 17.3 percentage point improvement from fine-tuning (34.3% vs. 17.0%) is comparable in relative terms to the improvement observed at the 355B scale (73.5% vs. 45.1%, a 28.4 percentage point gain), though the absolute performance remains substantially lower than the larger model. This suggests that the strategic curation methodology captures fundamental agentic patterns that transfer across model scales, consistent with the paper's claim that "our strategic data curation methodology captures fundamental patterns of agentic behavior that transfer effectively regardless of model capacity" (Section 4.4).

On generalization benchmarks (Table 3), LIMI-Air achieves 39.0% average, compared to GLM-4.5-Air at 33.2%—a 5.8 percentage point improvement that is smaller in both absolute and relative terms than the 14.2 percentage point gap between LIMI and GLM-4.5 on generalization (57.2% vs. 43.0%). This suggests that the smaller model benefits less from the curated training data on out-of-domain tasks, potentially because its lower base capacity limits how much it can generalize from 78 trajectories compared to the 355B model.

#### The Impact of CLI Environment Access

The dual-environment evaluation (Table 4 vs. Table 3) isolates the contribution of tool access to LIMI's performance:

Without CLI access (Table 4), LIMI achieves 50.0% average across generalization benchmarks (AgencyBench is not reported in this setting since its tasks require tool interaction). The base GLM-4.5 without CLI access achieves 48.7%, giving LIMI a small but consistent 1.3 percentage point advantage in the tool-free setting.

With CLI access (Table 3), LIMI achieves 57.2% average. The 7.2 percentage point gap between tool-free and tool-augmented evaluation (57.2% vs. 50.0%) represents the amplification from environmental access—the model's ability to leverage tools, execute code, search for information, and interact with external systems. The 1.3 percentage point gap between LIMI and GLM-4.5 in the tool-free setting represents the intrinsic reasoning improvement from fine-tuning, independent of tool-use skill.

The per-benchmark breakdown without CLI access (Table 4) shows:

- tau2-bench-airline: LIMI 40.0% vs. GLM-4.5 32.0% (+8.0 points)
- tau2-bench-retail: LIMI 49.1% vs. GLM-4.5 52.6% (−3.5 points)
- DS-1000: LIMI 54.8% vs. GLM-4.5 53.2% (+1.6 points)
- EvalPlus-HumanEval: LIMI 92.5% vs. GLM-4.5 92.1% (+0.4 points)
- EvalPlus-MBPP: LIMI 80.4% vs. GLM-4.5 79.6% (+0.8 points)
- SciCode-MP: LIMI 3.3% vs. GLM-4.5 3.3% (equal)
- SciCode-SP: LIMI 28.1% vs. GLM-4.5 27.8% (+0.3 points)

The most interesting finding here is the **reversal on tau2-bench-retail**, where GLM-4.5 (52.6%) outperforms LIMI (49.1%) without CLI access, despite LIMI's advantage with CLI access (45.6% vs. 36.8% in Table 3). This suggests that LIMI's training may have partially specialized the model toward tool-augmented interaction patterns on certain tasks, with a small cost to tool-free performance—a trade-off the paper does not discuss. The substantial gap on tau2-bench-airline (40.0% vs. 32.0%) without CLI access suggests that the intrinsic improvements are real but task-dependent, with some tasks benefiting more than others from the fine-tuning's effect on foundational reasoning.

### Ablation Studies and Robustness Checks

The paper contains remarkably few ablation studies relative to the scope of its claims. There is no systematic investigation of how performance varies with training set size, no comparison of different trajectory collection protocols, no analysis of the relative contribution of real-world versus PR-synthesized queries, and no sensitivity analysis of the key design choices (number of queries, trajectory collection strategy, choice of GPT-5 as trajectory generator). The following represents the complete set of what the paper provides as ablations and robustness checks:

**Model scale robustness**: The LIMI-Air vs. GLM-4.5-Air comparison (Table 2) shows that fine-tuning on the 78-sample dataset improves both 106B and 355B model variants. LIMI-Air achieves 34.3% vs. GLM-4.5-Air at 17.0%, a 17.3 percentage point improvement. This provides evidence that the data curation approach transfers across model scales, though the absolute performance gap relative to the larger model (73.5% vs. 34.3%) is substantial, indicating that pretraining scale remains important—the data efficiency principle applies to eliciting agentic capabilities from a capable base model, not to replacing the need for a capable base model entirely.

**Alternative dataset comparison**: The GLM-4.5-CC, GLM-4.5-Web, and GLM-4.5-Code comparisons (Tables 2 and 3) serve as the primary ablation validating the quality of LIMI's curated data over scale-based alternatives. The consistent underperformance of these alternatives—including the finding that GLM-4.5-Web and GLM-4.5-CC perform worse than the untuned GLM-4.5 on AgencyBench—provides evidence that training data quality, not just quantity, determines agentic fine-tuning outcomes. However, this is not a controlled ablation of the LIMI data construction methodology; it is a comparison against entirely different datasets collected under different protocols for different purposes. The performance gaps may reflect differences in task domain, interaction format, tool ecosystem, or data quality rather than the specific curation principles that LIMI advocates.

**Environment access ablation**: The with-CLI vs. without-CLI comparison (Tables 3 and 4) serves as an ablation of environment-dependence, showing that LIMI's improvements have both intrinsic (tool-independent) and extrinsic (tool-amplified) components. The 1.3 percentage point average advantage without CLI and 7.2 percentage point amplification with CLI provide evidence for the synergy principle discussed in Section 4.5.

**Cross-domain generalization**: The evaluation across tau2-bench, EvalPlus, DS-1000, and SciCode (Tables 3 and 4) serves as a domain generalization ablation, testing whether LIMI's improvements are specific to AgencyBench or transfer to other capability assessments. The consistent though modest improvements across most benchmarks (with the notable exception of tau2-bench-retail without CLI and SciCode-MP where performance is essentially identical) provide evidence for transfer, but the substantially smaller gains on out-of-domain benchmarks (average improvement of a few percentage points) compared to the dramatic gains on AgencyBench (28.4 percentage points) suggest that the improvements are partially domain-specific.

**Notable missing ablations**: The paper does not report:
- Performance at intermediate training set sizes (e.g., 10, 20, 40, 78, 150 samples) to characterize the scaling curve and determine whether 78 is optimal or whether further gains are possible with more data.
- Ablation of trajectory collection protocol (e.g., single-attempt trajectories vs. persist-until-success trajectories; human-collaborative vs. solo-agent trajectories; GPT-5-generated vs. GLM-4.5-generated trajectories).
- Ablation of query sources (real-world only vs. PR-synthesized only vs. the 60/18 mix).
- Ablation of trajectory length (do the longest trajectories contribute disproportionately, or would a set of shorter trajectories perform similarly?).
- Ablation of the Slime framework hyperparameters or comparison against alternative fine-tuning approaches.
- Multiple training runs with different random seeds to assess stability of the 78-sample training set.
- Human evaluation or inter-annotator agreement metrics for the quality assurance process during data construction.

### Critical Assessment

#### Claim: "Strategic curation of minimal high-quality demonstrations yields superior agentic intelligence compared to large-scale data accumulation" (the Agency Efficiency Principle)

**What the experiments demonstrate**: Table 2 shows that LIMI (78 samples) achieves 73.5% on AgencyBench while GLM-4.5-Code (10,000 samples) achieves 47.8%, a 53.7 percentage point advantage with 128× fewer samples. Table 3 shows that on generalization benchmarks, LIMI achieves 57.2% vs. GLM-4.5-Code at 40.9%.

**What the experiments do NOT demonstrate**: The comparison is not between two datasets constructed under the same methodology with different sizes. It is between a dataset constructed using a specific, carefully described curation protocol (real-world queries, human-GPT-5 collaborative trajectories, persist-until-success collection, SII CLI environment) and three entirely different datasets collected for different purposes (CC-Bench, AFM-WebAgent, AFM-CodeAgent) using different methodologies in different environments. The performance gap could reflect differences in task distribution, tool ecosystem compatibility with AgencyBench, trajectory format, data quality, or any combination of these factors—not necessarily the principle that curation beats scale per se. A fair test of the Agency Efficiency Principle would compare LIMI's 78-sample dataset against a random subset of 78 samples from the AFM-CodeAgent dataset, or against a scaled-up version of LIMI's own curation protocol (e.g., 500 or 1,000 similarly curated trajectories). Without this controlled comparison, the claim that the specific sample count (78) and the specific curation protocol are responsible for the performance advantage, rather than domain alignment between training and evaluation, remains unverified.

**Additional concern**: Two of the three alternative datasets produce **worse performance than the untuned base model** (GLM-4.5-Web at 36.7% and GLM-4.5-CC at 29.2% vs. GLM-4.5 base at 45.1%). This suggests that these datasets may be actively misaligned with AgencyBench's task structure—perhaps training on interaction patterns, tool conventions, or task types that conflict with what the benchmark requires. The fact that the base model outperforms fine-tuned variants on these datasets suggests that the comparison may be partly measuring negative transfer from misaligned data rather than positive transfer from curated data. If the alternative datasets were simply poorly suited to AgencyBench, the apparent advantage of curation would be inflated.

#### Claim: "Machine autonomy emerges not from data abundance but from strategic curation of high-quality agentic demonstrations"

**What the experiments demonstrate**: The dramatic performance difference between LIMI and the scale-based alternatives provides existence proof that strategic curation **can** yield superior results to less-curated large-scale data for the specific combination of GLM-4.5 base model, AgencyBench evaluation, and the alternative datasets tested.

**What the experiments do NOT demonstrate**: That machine autonomy **cannot** emerge from data abundance. The paper shows that one specific large-scale approach fails, not that all large-scale approaches would fail. A model trained on 10,000 similarly curated trajectories (rather than the AFM-CodeAgent dataset) might substantially outperform LIMI, demonstrating that abundance + quality beats quality alone. The paper's rhetorical framing ("fundamentally different development principles") implies a universality that the experiments do not test. The claim would require showing that scaling curated data produces diminishing or negative returns—a scaling curve that the paper does not provide.

**Additional concern**: The base model GLM-4.5 already achieves 45.1% on AgencyBench with **zero** agent-specific fine-tuning samples. The 78 trajectories add 28.4 percentage points. This means that 62% of LIMI's final performance (45.1/73.5) comes from the base model's pretraining on massive data, while 38% comes from the 78 curated trajectories. The paper emphasizes the 78 samples while the base model's contribution—itself the product of data abundance at pretraining scale—is the foundation on which those 78 samples build. The claim that "machine autonomy emerges not from data abundance" is therefore misleading: the autonomy emerges from a combination of pretraining-scale data abundance (providing language understanding, reasoning, and code generation capabilities) and fine-tuning-scale strategic curation (eliciting agentic application of those capabilities). The appropriate framing is that **eliciting** agency from a capable base model requires curation, not that agency itself is independent of data abundance.

#### Claim: "53.7% improvement over models trained on 10,000 samples—achieving superior agentic intelligence with 128 times fewer samples"

**What the experiments demonstrate**: This is arithmetically correct for the specific comparison between LIMI (78 samples, 73.5%) and GLM-4.5-Code (10,000 samples, 47.8%). The 128:1 ratio is accurately computed as 10,000/78 ≈ 128.2.

**What the experiments do NOT address**: The 10,000-sample AFM-CodeAgent-SFT-Dataset likely has a fundamentally different average trajectory length and structure from LIMI's trajectories. If AFM trajectories average significantly fewer tokens per sample—plausible given that LIMI's trajectories average 42.4K tokens with some reaching 152K tokens (Figure 4)—then the 128:1 sample ratio dramatically understates the ratio of total learning signal. A fairer comparison would report total training tokens or total agentic decision points. Without this information (the paper does not provide token counts for the alternative datasets), the 128:1 figure is potentially misleading in the opposite direction from the paper's intended message: the "efficiency" might actually be even greater than claimed if the alternative datasets have shorter trajectories, or smaller than claimed if they have comparably long trajectories. The paper's silence on this point prevents rigorous assessment of the data efficiency claim in terms that matter for practical training cost (total tokens processed, total GPU-hours consumed).

#### Claim: "Sophisticated agentic intelligence can emerge from minimal but strategically curated demonstrations of autonomous behavior"

**What the experiments demonstrate**: That fine-tuning on 78 trajectory examples substantially improves performance on AgencyBench and, to a lesser extent, on related benchmarks. This is a successful demonstration of few-shot supervised fine-tuning for a specific capability, consistent with a large body of work on instruction tuning and task-specific fine-tuning.

**What the experiments do NOT demonstrate**: "Emergence" in any strong sense. The model is not spontaneously developing agency from a small number of examples—it is being explicitly trained through supervised learning to imitate the specific behavior patterns demonstrated in the trajectories. The word "emerge" implies a qualitatively discontinuous transition (like the phase changes observed in pretraining scaling), but the experimental design does not test for this: there is no scaling curve showing a sharp performance transition at some critical dataset size, no comparison showing that 78 examples achieve something qualitatively different from what 50 or 100 examples would achieve, and no demonstration that the capability generalizes beyond the demonstrated behavior patterns to novel forms of agency not present in the training trajectories. What the experiments show is effective few-shot imitation learning, which, while valuable, is a different claim than emergence.

#### General methodological concerns

**Sample size and statistical reliability**: The evaluation uses 10 AgencyBench tasks and 6 generalization benchmarks with no reported confidence intervals, standard deviations, or multiple evaluation runs. The 78-sample training set is itself small enough that results could be sensitive to which specific queries are included. The paper does not report results across multiple different 78-query subsamples or with different random seeds for trajectory collection. Without such robustness checks, the reported point estimates may not be reliable indicators of expected performance.

**Single model family**: All fine-tuning experiments use GLM-4.5 and GLM-4.5-Air. The LIMI approach might interact with model architecture, pretraining data, or tokenizer characteristics in ways that do not transfer to other model families (e.g., Qwen, DeepSeek, Kimi, LLaMA). The paper acknowledges this implicitly by testing different GLM-4.5 scales but does not test on alternative architectures.

**Potential benchmark contamination**: The LIMI training data is constructed using a process that may share methodological similarities with AgencyBench construction (both involve multi-step collaborative tasks, both involve the SII CLI environment, both involve PhD annotators). The paper does not discuss whether there are any overlaps in the specific tasks, annotators, or design principles between the training data construction and the benchmark creation. Any such overlap would inflate LIMI's apparent performance relative to models not exposed to similar construction patterns during training.

**Missing training-inference trade-off analysis**: Unlike the compute-optimal scaling paper discussed in the reference example, LIMI provides no analysis of training cost. How many GPU-hours were required to generate the 78 trajectories using GPT-5 in SII CLI? How many GPU-hours for fine-tuning? How do these costs compare to training on 10,000 AFM-CodeAgent samples? Without this information, the data efficiency claim is about sample count rather than practical resource efficiency, and sample count can be a misleading proxy for cost when trajectory lengths vary by orders of magnitude.

**Missing difficulty analysis**: The paper does not analyze whether LIMI's improvements vary by task difficulty, query complexity, or trajectory length. It is plausible that the 78 curated trajectories are particularly effective for certain types of tasks (those closely matching the training distribution) and less effective for others, but the evaluation aggregates across all tasks without this breakdown. Such analysis would reveal the boundary conditions of the approach and whether, like the compute-optimal scaling paper, the efficiency principle is conditional on problem characteristics.

**The role of GPT-5**: The trajectories are generated by GPT-5, a model that is presumably significantly more capable than GLM-4.5. The fine-tuning is therefore distilling a stronger model's agentic behavior into a weaker model. The observed performance improvement may partially reflect the capability gap between GPT-5 and GLM-4.5 rather than the data curation principles per se. If trajectories were generated by a comparably capable model (e.g., GLM-4.5 itself, or another 355B-class model), would 78 trajectories still produce comparable improvements? The paper does not test this.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Unaccounted For — Making the 4× Efficiency Claim an Upper Bound

**The assumption or constraint**

The entire compute-optimal framework described in the paper rests on the ability to estimate prompt difficulty before deciding how to allocate the inference budget. The method for doing so—generating 2048 samples per question and averaging either ground-truth correctness (oracle) or PRM final-answer scores (predicted)—is extraordinarily expensive. The authors acknowledge this explicitly in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

At 2048 samples per question, the difficulty estimation step alone consumes more compute than the largest test-time budgets studied (256–512 generations). For the predicted difficulty approach, each of those 2048 samples must also be scored by the PRM, adding further computation.

**The consequence**

The reported 4× efficiency gains over best-of-N are computed after difficulty is already known, without amortizing the cost of learning it. In a realistic deployment, the total cost would be difficulty estimation + strategy execution, and the former could dominate the latter. For a deployment processing thousands of queries with an average budget of 16–64 generations, spending 2048 generations per query to estimate difficulty before allocating the remaining budget would make the total cost far higher than simply running best-of-256 on every query. The 4× figure is therefore best understood as an upper bound on achievable efficiency assuming a separate, cost-free difficulty oracle—not a realized deployment gain.

The paper's predicted difficulty approach partially addresses this by removing the need for ground-truth labels, but it does not reduce the generation cost: 2048 samples must still be generated and scored. The paper explicitly flags this as a key avenue for future work (Section 3.2: "future work on pretraining or finetuning models to directly predict difficulty of a question"), but no such model is developed or evaluated.

**What evidence exists in the paper**

Figures 4 and 8 show that compute-optimal scaling with predicted difficulty bins tracks oracle bins closely, particularly at lower budgets. However, these figures plot accuracy versus generation budget excluding the difficulty estimation cost. The paper does not provide any analysis that includes difficulty estimation in the budget calculation, nor does it report what fraction of total inference compute would be consumed by difficulty estimation at different deployment scales. There is no ablation showing how performance degrades if fewer than 2048 samples are used for estimation—a critical practical question that remains unanswered.

**Mitigation status**

The paper acknowledges this limitation explicitly (Section 3.2) and suggests future work on training models to predict difficulty directly from question text, which would eliminate the sampling cost entirely. The authors frame the current approach as an exploration-exploitation tradeoff that future work could optimize. However, no mitigation is implemented in the current system, and the difficulty estimation cost is entirely externalized from all efficiency calculations. This is a significant gap between the paper's conceptual contribution and its practical deployability.

---

### 6.2 All Results Are from a Single Benchmark and a Single Model Family

**The assumption or constraint**

Every experiment in the paper uses the MATH benchmark (500 test questions) with PaLM 2-S* as the base model. The paper does not evaluate on any other reasoning benchmark (GSM8K, MMLU-Math, TheoremQA), any code generation benchmark (HumanEval, MBPP), or any model family other than PaLM 2. The authors state that they "believe this model is representative of the capabilities of many contemporary LLMs" (Section 4), but this claim is unverified.

**The consequence**

Without replication on other benchmarks or model families, the paper's findings—the difficulty-dependent behavior of search algorithms, the 4× efficiency gains from compute-optimal allocation, the PRM's superiority over ORMs with last-step aggregation, the FLOPs-matched comparison results—may be specific to PaLM 2-S*'s particular output distribution, calibration properties, and error patterns, or to MATH's particular task structure (competition-level math problems with clean final answers). Several aspects of the results could plausibly not transfer:

- The PRM's over-optimization behavior (beam search degrading easy-problem performance at high budgets, Figure 3 right) depends on the PRM's calibration properties, which are a function of both the base model's output distribution and the Monte Carlo rollout training procedure. A model with different error patterns might show over-optimization at different budget levels or on different difficulty tiers.
- The revision model's ability to learn from in-context examples depends on the base model's in-context learning capabilities, which vary substantially across model families (some models are much better at leveraging in-context demonstrations than others).
- The difficulty bins are defined relative to PaLM 2-S*'s pass@1 distribution. A model with substantially higher or lower base performance on MATH would produce different bin boundaries, potentially shifting which strategies are optimal at which difficulty levels.
- The MATH benchmark consists exclusively of problems with exact, verifiable answers amenable to string matching. It is unclear whether the difficulty-dependent patterns generalize to reasoning domains with fuzzier correctness criteria (code generation where correctness requires execution, scientific QA where multiple answers may be acceptable, open-ended generation tasks).

**What evidence exists in the paper**

The paper provides no cross-benchmark or cross-model evaluation. All figures, tables, and analyses draw from the same 500-question MATH test set and PaLM 2-S* model. The FLOPs-matched comparison uses only one alternative model (the ~14× larger variant). There is no evidence, positive or negative, about how the findings would transfer.

**Mitigation status**

The paper acknowledges the single-benchmark limitation implicitly by not claiming universality, but it does not explicitly discuss it as a limitation or propose cross-validation on other benchmarks as future work. The authors' belief that PaLM 2-S* is "representative" is stated as an assumption rather than a testable claim. This is a significant gap for practitioners considering whether to adopt compute-optimal test-time strategies: without knowing whether the difficulty-dependent patterns observed here hold for other model families (e.g., LLaMA, GPT, Claude, Qwen) and other task types (code generation, logical reasoning, scientific QA), the generalizability of the approach remains an open question.

---

### 6.3 The 14× Larger Model Baseline Is Not Compute-Optimally Trained — Making the Training-Inference Tradeoff Potentially Overstated

**The assumption or constraint**

The FLOPs-matched comparison in Section 7 scales model parameters while holding training data fixed, following the LLaMA paradigm (Touvron et al., 2023) where data quantity is not increased proportionally with model size. The authors acknowledge this explicitly:

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work."

This departs from the Chinchilla scaling laws (Hoffmann et al., 2022), which established that for compute-optimal pretraining, model size and data quantity should be scaled approximately equally. A Chinchilla-optimal model trained with ~14× more total pretraining FLOPs would be smaller in parameter count but trained on proportionally more data, potentially achieving better performance than a parameters-only-scaled model.

Additionally, the ~14× larger model uses only greedy decoding with no test-time compute augmentation. It is never evaluated with majority voting, best-of-N, or any of the search or revision strategies that the smaller model receives.

**The consequence**

The reported advantages of test-time compute over pretraining—for example, +27.8% on medium questions at the low inference-to-pretraining ratio regime (Figure 1, top-right bar chart)—may be inflated relative to what a fair comparison would show. If the larger model were compute-optimally trained (scaling both parameters and data) and given even a modest test-time compute budget (e.g., best-of-8), the performance gap might shrink substantially or reverse in some regimes.

The paper's key takeaway—"test-time compute can substitute for pretraining on easy-to-medium problems"—is therefore qualified by an important methodological choice: the pretraining baseline is weaker than it needs to be. A practitioner deciding between training a larger model or investing in test-time compute infrastructure needs to know whether the substitution effect holds against the best possible pretrained model they could build with the same total budget, not against a parameter-scaled model that may be undertrained relative to compute-optimal standards.

**What evidence exists in the paper**

Section 7 and Figure 9 present the FLOPs-matched comparison with the parameters-scaled baseline. The paper does not provide any comparison against a Chinchilla-optimal baseline, nor does it include an ablation where the larger model is given any test-time compute budget (the larger model always uses greedy decoding). The paper also does not report the absolute performance of the ~14× larger model on MATH in standard evaluation settings, making it impossible to assess whether this model is indeed stronger than PaLM 2-S* or whether the parameter scaling conferred benefits that data scaling might have matched more efficiently.

**Mitigation status**

The paper acknowledges the non-Chinchilla-optimal baseline and frames the Chinchilla-optimal comparison as future work. The authors are transparent about this choice and the reasons for it (the LLaMA paradigm is representative of common practice). However, the paper does not discuss the potential magnitude of the bias introduced by this choice, nor does it provide bounding analysis suggesting how results might change under compute-optimal pretraining. A practitioner reading the paper might reasonably conclude that test-time compute is broadly preferable to pretraining for easy-to-medium problems, without realizing that this conclusion depends on a pretraining baseline that is known to be suboptimal.

---

### 6.4 The Hardest Problems Remain Essentially Unsolved — Defining a Hard Capability Ceiling

**The assumption or constraint**

Across all methods evaluated in the paper—best-of-N, beam search, lookahead search, compute-optimal search, sequential revisions, and hybrid sequential-parallel revision strategies—performance on the hardest difficulty quintile (bin 5) remains near zero regardless of compute budget. The base model's pass@1 on these problems is already close to zero, and no test-time strategy compensates.

**The consequence**

Test-time compute can amplify existing capability but cannot create capability from nothing. If the base model's probability of producing a correct solution is essentially zero on a given problem class, no amount of search, sampling, or revision will find a correct solution—there are simply no correct candidates in the proposal distribution to discover or refine. This is a fundamental limitation that the paper is candid about (Section 7 takeaway box), but it has significant practical implications:

- For deployment scenarios where the query distribution includes a substantial fraction of hard problems, test-time compute strategies offer essentially no path to improvement—the only option is to improve the base model through pretraining.
- The compute-optimal allocation framework does not provide a solution for hard problems; it merely identifies them as beyond reach and presumably allocates minimal budget to them (though the paper does not report budget allocation per difficulty bin explicitly).
- The difficulty bins are model-relative—bin 5 for one model might be bin 3 for a stronger model. As base models improve, the set of problems in each bin shifts, and what is currently "hard" becomes "medium" or "easy." However, there will always be some problems in bin 5 (the hardest 20% relative to the model's capability distribution), and these problems will remain unsolved by test-time compute for any given model generation.

**What evidence exists in the paper**

Figure 3 (right panel) shows bin 5 accuracy hovering at 1–3% for all search methods and all budget levels up to 256 generations. Figure 7 (right panel) shows bin 5 accuracy at roughly 2–3% regardless of sequential-to-parallel ratio, with no meaningful variation. Figure 9 shows the bin 5 scaling curve (blue, bottommost) essentially flat near 0–5% for both revisions and PRM search, well below the ~14× larger model's performance on the same problems. The FLOPs-matched comparison (Figure 1) reports relative disadvantages of −37.2% to −52.9% for hard problems when using test-time compute instead of the larger model, at high inference-to-pretraining ratios.

**Mitigation status**

The paper is transparent about this limitation. The Section 7 takeaway explicitly states that for hard problems, "pretraining is almost always more effective." The limitation is not framed as a weakness of the approach but rather as a boundary condition: test-time compute scaling is most effective when problems are within the base model's capability range, and it cannot compensate for fundamental capability gaps. The paper does not propose any mitigation (since the limitation is inherent to the proposal distribution framework—if correct solutions are not in the distribution, no amount of test-time selection or refinement can find them). The clear articulation of this boundary is a strength of the paper's analysis, even though it constrains the practical applicability of the method.

---

### 6.5 Revisions and Search Are Studied Independently — the Full Potential of Combined Strategies Is Untested

**The assumption or constraint**

The paper studies two complementary mechanisms for improving test-time performance—PRM-guided search (which modifies how outputs are selected from a fixed proposal distribution) and iterative revisions (which modifies the proposal distribution itself so that better candidates are generated)—but never combines them in a single experiment. Section 8 explicitly acknowledges this:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

While the paper frames revisions and search as complementary axes within a unified framework (Section 2), the experimental design treats them as independent alternatives. The compute-optimal policy selects between search strategies (Section 5) or selects sequential-to-parallel ratios for revisions (Section 6), but never jointly optimizes over both.

**The consequence**

The paper's results represent a lower bound on what a fully integrated system could achieve. There are several natural combinations that could yield gains beyond either mechanism alone:

- Using the revision model as the proposal distribution within beam search: at each step of the search tree, the model conditions on the previous rejected branches as revision context, potentially producing higher-quality candidate steps informed by the search history.
- Using the PRM to guide which revisions to pursue: rather than generating a long sequential revision chain blindly, use the PRM's per-step scores to decide when a revision is on the right track (continue) versus when the current trajectory has derailed and should be restarted from scratch.
- Combining the difficulty-dependent allocation policies for search and revisions into a single meta-policy that selects not just between search algorithms or sequential-parallel ratios, but between the entire space of strategies (best-of-N, beam search, revisions-only, hybrid search+revision) based on the estimated difficulty.

Without these experiments, the paper leaves open the question of how much additional gain could be achieved by integrating the two mechanisms. The 4× efficiency improvements over best-of-N, already significant, might represent only a fraction of the total potential gains accessible through full integration.

**What evidence exists in the paper**

No empirical evidence addresses the combined approach. The paper provides separate results for search (Section 5, Figures 3–4) and revisions (Section 6, Figures 6–8) but does not include any experiment where both mechanisms operate simultaneously on the same problem. The Section 8 discussion explicitly flags this as an avenue for future work, suggesting that the authors see the combination as promising but unexplored.

**Mitigation status**

The paper acknowledges this as future work but provides no preliminary analysis, bounding argument, or hypothesis about the magnitude of potential combined gains. Given that the paper's core contribution is a framework for understanding test-time compute allocation, the absence of combined experiments is a significant gap—the natural endpoint of the framework is a unified policy that deploys both proposal modification and verifier-guided selection adaptively, but the paper stops short of demonstrating this. A practitioner wanting to maximize test-time performance would need to experimentally determine the optimal combined strategy themselves, as the paper provides no guidance on the interaction between the two mechanisms.

---

### 6.6 The Test Set Is Small (500 Questions) and Difficulty Bins Split It into Critically Small Subgroups

**The assumption or constraint**

All experimental results are derived from the MATH test set of 500 questions. Within the per-difficulty-bin analyses that form the core of the compute-optimal allocation framework, these 500 questions are split into five quintiles of approximately 100 questions each. Two-fold cross-validation further splits each bin roughly in half for strategy selection versus evaluation, meaning compute-optimal policies are selected based on approximately 50 questions per fold per bin.

**The consequence**

With only ~50 questions per bin for strategy selection, the compute-optimal policy is estimated from a very small sample. The variance of the policy selection process could be substantial: the strategy that appears optimal for a given difficulty-budget pair based on 50 questions may not be the true optimal strategy for that difficulty level, and small differences in which specific questions fall into each fold could produce different policy selections. The paper does not report any measure of variability—no confidence intervals on the compute-optimal scaling curves, no error bars on per-bin accuracies, no analysis of how stable the policy selections are across different random splits.

This has direct implications for the paper's central efficiency claims. If the 4× improvement figure is sensitive to the specific random split used for cross-validation, or to which 500 questions constitute the test set, then the headline result may not be robust. A practitioner trying to replicate the approach would need to determine the optimal strategies empirically for their own model and problem distribution, and the paper provides no guidance on how much data is needed for reliable strategy selection at a given difficulty granularity.

Additionally, for the hardest difficulty bin (bin 5, ~100 questions), the near-zero performance across all methods (1–3% in Figures 3 and 7) is estimated from a sample where a single correctly answered question represents roughly 1% accuracy. The apparent flatness of the bin 5 curves could reflect genuine impossibility (no method helps) or could reflect insufficient statistical resolution to detect small but real improvements—with only ~100 questions, a method that improves bin 5 accuracy from 2% to 5% (a 2.5× relative improvement) would represent only 3 additional correct answers, which could easily be masked by sampling noise.

**What evidence exists in the paper**

The paper reports point estimates without confidence intervals throughout. Figure 4 shows compute-optimal search curves for oracle and predicted bins that "largely overlap," and Figure 8 shows similar overlap for compute-optimal revisions, but neither figure provides any quantification of variability. The cross-validation protocol is described in Section 3.2, but the paper does not report how consistent the policy selections are across folds, what fraction of difficulty-budget pairs have unambiguous best strategies versus ties, or how sensitive the aggregate performance is to the specific fold assignment.

**Mitigation status**

The paper addresses the circularity problem (not optimizing and evaluating on the same data) through two-fold cross-validation, which is a reasonable methodological choice given the small test set. However, the paper does not address the statistical reliability problem—the cross-validation folds are still only ~50 questions each per bin, and the paper provides no evidence that the selected policies are stable. There is no discussion of statistical significance, confidence intervals, or required sample sizes for reliable policy selection. This is a gap that limits the strength of the conclusions that can be drawn from the paper's quantitative results, particularly the specific efficiency ratios (4×) and the precise shapes of the compute-optimal scaling curves.

## 7. Implications and Future Directions
- How this changes the landscape
  - If agentic capabilities scale with quality and completeness of trajectories rather than dataset size, the field should redirect effort from mass data synthesis to strategic curation and environment design. This reframes the “agent scaling law” from volume-centric to information- and structure-centric (Abstract; Sections 1, 6).

- Follow-up research enabled/suggested
  - Token- and step-normalized scaling studies: quantify “less is more” as a function of tokens, states visited, and unique tool interactions to formalize an “agency information law.”
  - Cross-domain replication: apply LIMI to other agent domains (e.g., data engineering, bioinformatics, robotics control) and to different base model families.
  - Ablations within LIMI data: vary the number of queries, trajectory length, success vs. failure traces, and degree of human intervention to identify the most causally important ingredients.
  - Beyond SFT: combine curated trajectories with preference optimization or offline RL tailored to long-horizon POMDP settings (Section 5.1 discussion), and test whether LIMI-style curation complements RL.

- Practical applications
  - Enterprise coding assistants that can own tickets end-to-end (triage → implement → test → PR) using curated development traces.
  - Research copilots that execute full workflows (dataset search, evaluation, statistical analysis, report generation) as in Tasks 5–7 and Task 6’s metric suite (Table 1; pages 20–22).
  - Education and training: use LIMI’s trajectory structure to teach planning and tool use in safe sandboxes before deployment.

> Most central quantitative takeaway (Table 2; Figure 1): “LIMI reaches 73.5% on AgencyBench with only 78 curated samples,” outperforming strong baselines and models trained on 7.6k–10k-agent datasets. The curated, long-horizon trajectories—capturing reasoning, tool calls, and environment feedback—appear to be the key driver of agentic competence.
