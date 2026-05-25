# Attention Heads of Large Language Models: A Survey

**ArXiv:** [2409.03752](https://arxiv.org/abs/2409.03752)

## 🎯 Pitch

This paper delivers the first comprehensive, up-to-date survey focused specifically on the functional roles of attention heads in modern decoder-only large language models. By introducing a novel, cognitively inspired four-stage framework—spanning knowledge recall to expression—and unifying discovery methods and benchmarks, the authors demystify how attention heads cooperate to produce LLMs’ remarkable behaviors. This work matters because unlocking the mechanisms and functions of attention heads not only advances scientific understanding, but also equips researchers with concrete tools to make LLMs more accurate, transparent, and steerable in practical settings.

---

## 1. Executive Summary
This survey systematizes what is known about “special” attention heads inside modern decoder‑only large language models (LLMs) and proposes a four‑stage, cognitively inspired framework—Knowledge Recalling (KR), In‑Context Identification (ICI), Latent Reasoning (LR), and Expression Preparation (EP)—to explain how such heads cooperate to produce answers (Section 4; Figures 6–8). It also unifies discovery methods (activation patching, ablations, probing, scoring, simplified models) and compiles evaluation datasets and metrics so researchers can identify, test, and compare functional attention heads (Section 5; Tables 4–5; Section 6; Tables 6–7).

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs are high‑performing but remain “black boxes.” The paper targets the specific internal components—attention heads—that appear to implement distinct sub‑skills such as copying names, doing pattern induction, or amplifying correct choices (Section 4; Figure 7).
  - There is no up‑to‑date, LLM‑focused survey of attention head mechanisms; prior surveys either emphasized non‑Transformer architectures, early attention variants, or broad interpretability methods without consolidating what each head type actually does in today’s LLMs (Section 3.3).

- Why it matters
  - Understanding head functions enables principled interventions to reduce errors (e.g., hallucinations), improve truthfulness and consistency, and steer models at inference time (Sections 1, 4.4.2, 5.1; also see heads like Truthfulness/Accuracy/Consistency in Figure 7).
  - Insights support theory building (e.g., circuits, residual streams) and practical tooling (e.g., KV‑cache compression via retrieval heads; Figure 7 and citations 69–70).

- Prior approaches and gaps
  - Early interpretability focused on BERT‑style encoders or on many attention variants that are no longer central to mainstream LLMs (Section 3.3).
  - Mechanistic studies identified individual circuits (e.g., IOI—Indirect Object Identification) mostly in small models like GPT‑2 Small, but lacked a unifying cognitive framework and did not synthesize discovery methods and benchmarks in one place (Section 4.6; Figure 9).

- Positioning
  - Provides a mathematical “wiring diagram” for decoder‑only Transformers (Section 3.1; Equations 1–4; Figure 3) and key conceptual tools—`residual streams`, `QK` and `OV` matrices, and `circuits` (Section 3.2; Figure 4).
  - Organizes known head types by stage of reasoning (Figure 7), maps stages to layer depth (Figure 8), and shows collaborative patterns across heads with worked examples (Section 4.6; Figure 9).

## 3. Technical Approach
This is a survey, but it offers a precise technical scaffolding for understanding how attention heads work and how to study them.

A. Model anatomy and notation (Section 3.1; Equations 1–4; Figure 3)
- A decoder‑only LLM consists of an embedding layer, L Transformer blocks, and an unembedding layer. Each block has:
  - Multi‑head attention: outputs from H heads are summed and residual‑added to the input (Equation 1).
  - Feed‑Forward Network (FFN/MLP): its output is residual‑added to produce the next block’s input (Equation 2).
- For head `h` in layer `ℓ`, queries/keys/values are `Q^h_ℓ = X W^Q`, `K^h_ℓ = X W^K`, `V^h_ℓ = X W^V`; attention computes `softmax(Q K^T) V O` (Equation 3).
- Expanding this shows two compound matrices (Equation 4):
  - `QK matrix = W^Q W^{K⊤}` determines where a head looks (which tokens/timesteps).
  - `OV matrix = W^V O` determines what a head writes back to the residual stream.

B. Conceptual tools for mechanism tracing (Section 3.2; Figure 4)
- Residual streams: every token position carries a running sum of prior computations; heads read from and write to this shared highway, enabling cross‑layer and cross‑token information flow.
- Circuits: subgraphs of interacting components (heads, FFNs) that implement tasks (e.g., bias circuits, knowledge circuits).
- Logit lens: project intermediate vectors through the unembedding to approximate token‑level preferences; useful to quantify effects of interventions.

C. Four‑stage cognitive framework (Section 4; Figures 6–8)
- Stages are not strictly linear; reasoning can loop between stages (Figure 6).
  1) Knowledge Recalling (KR): retrieve relevant stored knowledge or biases from parameters.
  2) In‑Context Identification (ICI): locate and transform structural, syntactic, and semantic cues in the prompt/history.
  3) Latent Reasoning (LR): integrate evidence and perform implicit computation (pattern induction, comparison, arithmetic/logical steps).
  4) Expression Preparation (EP): aggregate and amplify the result into tokens that the unembedding/softmax will emit.
- Typical layer mapping (not absolute): KR in shallow/middle layers; ICI spans shallow→deep; LR in middle→deep; EP deeper layers (Figure 8).

D. Taxonomy of head functions by stage (Figure 7; Sections 4.2–4.5)
Below are representative head types, how they work, and where they fit.

1) KR: heads that initialize or bias the reasoning
- Associative/Memory Heads (Section 4.2): treat weights like associative memories that denoise superposed activations; they recall attributes about entities surfaced by FFNs and write this back to the stream (citations 59–61).
- Task‑bias heads in special settings:
  - Constant/Single‑Letter Heads for multiple‑choice QA (MCQA): spread or focus attention over option letters to “collect” candidate answers before reasoning (Section 4.2; Table 2).
  - Negative Head for binary decision tasks: shows a pre‑learned bias toward negative answers by allocating more attention to “No”‑like tokens (Section 4.2; Figure 7; Table 2).

2) ICI: heads that parse structure and meaning from the context
- Structural heads (Section 4.3.1):
  - Previous/Positional Heads: encode previous‑token relations and positional patterns.
  - Rare Words / Duplicate Heads: attend to low‑frequency or repeated tokens to highlight salience.
  - (Global) Retrieval Heads: track specific mentions deep in long contexts—crucial for “needle‑in‑a‑haystack” retrieval (Figure 7; citations 69–70).
- Syntactic heads (Section 4.3.2):
  - Subword Merge: unify split word pieces into coherent units.
  - Mover/Name‑Mover/Backup/Negative‑Name‑Mover: copy important arguments (e.g., names) to the current decoding position ([END]) or suppress a copy when inappropriate (Figure 7).
- Semantic heads (Section 4.3.3):
  - Context and Content‑Gatherer Heads: move answer‑relevant tokens to [END]/[SUM] to stage evidence (Figure 7).
  - Sentiment Summarizer: aggregates sentiment‑bearing adjectives/verbs around [SUM] (Section 4.3.3).
  - Subject/Relation Heads; Semantic Induction Heads: extract entities and relations (Figure 7).

3) LR: heads that compute or decide
- In‑Context Learning (Section 4.4.1):
  - Task Recognition: a Summary Reader head reads [SUM] to map a described task to known labels (e.g., positive/negative).
  - Task Learning: Induction Heads detect patterns like “… A B … A → predict B” by matching “previous token” features from a Previous Head with current tokens (Section 4.4.1).
  - In‑Context Heads with metric‑learning flavor compute similarity between [END] representation and label prototypes to choose a label among several (Section 4.4.1).
- Effective reasoning property heads (Section 4.4.2):
  - Truthfulness/Accuracy/Consistency Heads correlate with truthful, correct, and self‑consistent outputs; steering along their directions can improve behavior.
  - Vulnerable Heads overreact to distractors; reducing their influence can improve robustness.
- Task‑specific LR (Section 4.4.3):
  - Correct‑Letter Head bridges textual answers to option letters in MCQA.
  - Iteration Head performs step‑by‑step state updates (e.g., parity or sequence iteration; Section 4.6).
  - Successor Head implements “+1” on ordinal numbers.
  - Inhibition/Suppression Head reduces the logits of disallowed candidates (e.g., suppress “John” in IOI; Section 4.4.3).

4) EP: heads that “package” the result for emission (Section 4.5; Table 3)
- Mixed Head aggregates outputs of Subject/Relation/Induction heads into a concise final vector.
- Amplification/Correct Heads boost the correct token(s) near [END] so the unembedding/softmax selects them.
- Coherence Head aligns generated language with the desired output language; Faithfulness Head improves consistency between internal reasoning and chain‑of‑thought text.

E. Collaboration patterns and circuits (Section 4.6; Figure 9)
- MCQA example: Content‑Gatherer at ICI moves answer text to [END]; Correct‑Letter at LR matches a “query” that asks “are you the correct label?” against keys that encode option letters plus their textual descriptions.
- Parity example: a Mover Head sends [EOI] to [END]; an Iteration Head finds the last digit and combines with the previous parity state to compute the final parity (Equation 5).
- IOI circuit (Figure 9): Subject/Relation Heads cue “answer should be a human name” (KR), Duplicate and Name‑Mover Heads collect candidate names (ICI), Induction plus Previous Heads propagate “John” salience while an Inhibition Head suppresses “John” (LR), and an Amplification Head boosts “Mary” at EP.

F. Methods to discover and validate head functions (Section 5; Tables 4–5; Figure 10)
- Modeling‑Free
  - Modification‑based: directional addition/subtraction assumes some concept direction in representation space; e.g., add a “positive‑minus‑negative” sentiment vector at a head to test if it summarizes sentiment (Section 5.1).
  - Replacement‑based: zero/mean ablation or naïve activation patching (replace an activation from the clean prompt with one from a corrupted prompt, or vice versa) to see which heads matter (Section 5.1).
- Modeling‑Required
  - Training‑Required: probing (train classifiers on head activations to identify functions); simplified model training (train tiny Transformers on clean tasks to reveal mechanisms more clearly) (Section 5.2).
  - Training‑Free: scoring metrics such as Retrieval Score (Equation 6) for retrieval heads and Negative Attention Score (NAS; Equation 7) for negative‑bias heads; Information Flow Graphs to extract high‑impact edges across tokens and components (Section 5.2).

G. Evaluation resources (Section 6; Tables 6–7; Figure 11)
- Mechanism exploration datasets distill tasks into token‑level probes (e.g., IOI, ToyMovieReview/MoodStory with templates in Figure 11, Induction/Iteration/Succession tasks).
- Common evaluations test whether steering heads improves real‑world capability (MMLU, TruthfulQA, LogiQA, SST/SST‑2, long‑context retrieval, etc.).

## 4. Key Insights and Innovations
1) A cognitively grounded four‑stage framework for LLM reasoning (Section 4; Figures 6–8)
   - Innovation: frames head activity as cycles across KR→ICI→LR→EP, mirroring human problem solving modules (knowledge retrieval, perception/parse, reasoning, articulation).
   - Significance: clarifies how different heads cooperate and why some heads recur across tasks (e.g., Induction with Previous, Inhibition with Mover) rather than treating heads as isolated curiosities.
   - Difference from prior work: earlier studies documented specific circuits (e.g., IOI) but lacked a unifying, stage‑wise map that spans most observed head types.

2) A comprehensive taxonomy of special heads with concrete mechanisms (Figure 7; Sections 4.2–4.5)
   - Innovation: places dozens of reported heads into functional families with brief operational descriptions and links to where they were found (e.g., LLaMA, GPT, Pythia, Mistral).
   - Significance: provides a lookup for practitioners to hypothesize which heads to inspect or steer for a given failure mode (e.g., long‑context errors → Retrieval Heads).

3) Unification of discovery methodologies by dependency on modeling and manipulation type (Section 5; Tables 4–5; Figure 10)
   - Innovation: splits methods into modeling‑free vs modeling‑required and further into modification‑ vs replacement‑based (for the former) and training‑required vs training‑free (for the latter).
   - Significance: helps choose the right tool for a hypothesis—for example, when labels are unavailable, use training‑free scores like Retrieval Score/NAS; when controllability matters, train simplified models.

4) Concrete collaboration narratives and layer‑stage mapping (Sections 4.6; Figures 8–9)
   - Innovation: shows how sequences of heads produce end‑to‑end behavior in worked examples (parity, IOI, MCQA), and provides a typical mapping of stages to layer depth.
   - Significance: encourages multi‑head, circuit‑level analysis instead of single‑head anecdotes; aids debugging by predicting where in the stack to intervene.

5) Curated evaluation suites with prompt templates and equations (Section 6; Tables 6–7; Figure 11; Equations 6–7)
   - Innovation: compiles mechanism‑targeted datasets and introduces head‑specific scores.
   - Significance: supports reproducible comparison of head hypotheses and their downstream impact (e.g., sentiment templates in Figure 11 to find Sentiment Summarizers).

## 5. Experimental Analysis
Note: This is a survey; it does not run new experiments. Instead, it collates how the community evaluates head mechanisms and provides formulas and templates.

- Evaluation methodology (Section 6)
  - Mechanism‑focused datasets (Table 6) reduce tasks to token‑level probes so head effects can be cleanly measured:
    - IOI (indirect object identification), ICL‑MC for induction, Succession for ordinal “+1,” Iteration‑Synthetic for iterative state updates, ToyMovieReview/MoodStory for sentiment (Figure 11 templates), World‑Capital and LRE‑1 for knowledge recall.
  - Common capability benchmarks (Table 7) check whether steering heads improves global performance: MMLU (knowledge reasoning), TruthfulQA (truthfulness), SST/SST‑2 and ETHOS (sentiment/abuse), Needle‑in‑a‑Haystack (long‑context retrieval), AG News/TriviaQA/AGENDA (comprehension/generation).

- Metrics and instruments
  - Logit lens to quantify the effect of a head intervention at intermediate layers (Section 3.2.2).
  - Retrieval Score (Equation 6) to measure how reliably a head points to the intended token across examples.
  - Negative Attention Score (NAS; Equation 7) to quantify negative bias across “Yes/No” positions.

- Interventional methods and ablations (Section 5; Table 4)
  - Directional addition/subtraction to test linear concept directions (e.g., positive–negative sentiment direction added to a head’s activation).
  - Zero/mean ablation or naïve activation patching to identify necessity/sufficiency of a head’s activation for an observed behavior.

- Representative collaboration evidence
  - IOI circuit diagram (Figure 9) integrates Subject/Relation (KR), Duplicate/Name‑Mover (ICI), Induction/Previous and Inhibition (LR), and Amplification (EP). The diagram encodes paths of information flow rather than numeric effect sizes, but it summarizes replicated findings across studies on GPT‑2 (Section 4.6).
  - Parity and MCQA mini‑case studies describe the step‑by‑step information routing and matching queries vs keys (Section 4.6).

- Quantitative results
  - This survey aggregates methods and phenomena but does not tabulate numeric scores or effect sizes across models or tasks. Where numbers matter (e.g., “how much does steering a Truthfulness Head improve TruthfulQA?”), readers must consult the cited studies (Figure 7 citations 86–89). Within this paper, the quantitative pieces are definitions of scores (Equations 6–7) and setup specifics (layer ranges in Figure 8), plus a trend illustration (Figure 1) showing rising Google search interest in “attention head” and “model interpretability”.

- Do the summarized experiments support the claims?
  - Yes with caveats: the paper grounds mechanisms in multiple, often replicated case studies (e.g., Induction Heads64,80–85; Name‑Mover/Copy‑Suppression22,73; Retrieval Heads69–70) and shows converging evidence from activation patching, ablation, and simplified models. However, many demonstrations are in smaller models (e.g., GPT‑2 Small) and toy settings; transfer to frontier LLMs and open‑ended tasks is less documented (Section 8.1).

## 6. Limitations and Trade-offs
- Generalizability across tasks (Section 8.1)
  - Circuits validated on IOI, Color‑Object, or toy arithmetic may not directly map to open‑ended QA, math proofs, or tool‑use workflows.
- Transferability across model families (Section 8.1; Figure 7)
  - Many head types are reported in limited model series (e.g., GPT‑2, Pythia, LLaMA); whether the same head indices or even the same functions exist in other architectures is underexplored.
- Multi‑head collaboration under‑specified (Section 8.1)
  - Most studies isolate single heads; few provide complete, quantitative circuit decompositions across layers and tokens for complex tasks.
- Theoretical foundations (Section 8.1)
  - Evidence is largely empirical and interventional. There is no formal proof that the proposed circuits are necessary/unique; alternate mechanisms may implement the same behavior.
- Stage mapping is heuristic (Figures 6–8)
  - The KR→ICI→LR→EP sequence is helpful but not strict; models can revisit KR/ICI late in the stack, and simple tasks may skip EP entirely (Section 4.5). This blurs boundaries when classifying heads that appear at multiple depths.
- Computational constraints
  - Fine‑grained patching/ablation over all heads and positions in modern LLMs is expensive; some methods (probing, simplified model training) require additional data or training (Section 5.2).
- Measurement bias
  - Scores like NAS (Equation 7) depend on prompt formatting and choice of positions for “Yes/No”; conclusions about bias or vulnerability can be prompt‑sensitive (Sections 4.4.2, 5.2).

## 7. Implications and Future Directions
- How this changes the field
  - Shifts attention from “what a head looks at” to “what role it plays in a reasoning pipeline,” enabling circuit‑level debugging and targeted steering (Figures 7–9).
  - Encourages standardized, mechanism‑first evaluation: use token‑level probes and scores to validate a head hypothesis before deploying steering to real tasks (Section 6).

- Follow‑up research enabled/suggested (Section 8.2)
  - Complex tasks: extend circuit analysis to open‑ended QA, mathematical problem solving, and tool‑use pipelines where KR/ICI/LR/EP loops are longer and more nested.
  - Prompt‑robust mechanisms: study how head roles shift with paraphrases or instruction changes and design interventions that stabilize desired circuits.
  - New experimental tooling: automated circuit discovery (e.g., scalable Information Flow Graphs), causal tests of mechanism indivisibility/necessity, and better logit‑lens calibrations.
  - Cross‑model transfer: align head functions across families (e.g., LLaMA↔Mistral↔GPT) via feature‑space matching or causal abstraction to learn “universal” head archetypes.
  - Machine Psychology linkage (Section 7.2): formalize analogies to human cognition (e.g., metacognition/self‑feedback, working memory) to hypothesize new head roles and design behavioral tests.

- Practical applications
  - Reliability steering: boost Truthfulness/Accuracy/Consistency Heads or damp Vulnerable Heads during inference to reduce hallucinations and inconsistencies (Section 4.4.2).
  - Retrieval and efficiency: exploit Retrieval Heads for long‑context factuality and KV‑cache compression (Figure 7; citations 69–70).
  - Multi‑choice and grading: use Correct‑Letter/Amplification/Coherence Heads to improve MCQA selection and multilingual response consistency (Sections 4.4.3, 4.5).
  - Knowledge editing and conflict resolution: identify Memory/Context/Subject/Relation heads to localize and patch conflicting facts (Figure 7; citations 24, 61, 78).

In sum, this survey provides a structured map of attention‑head roles in LLMs, practical methods to discover and evaluate them, and a cognitively motivated lens for reasoning about model internals. The roadmap in Sections 8.1–8.2 highlights the main open problems: scaling circuit analysis to complex tasks and frontier models, formalizing theory, and building robust, transferable interventions.
