# Language Models are Few-Shot Learners

**ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165)

## 🎯 Pitch

This paper introduces GPT-3, a 175-billion-parameter language model that achieves remarkable 'in-context' learning: it performs a wide array of NLP tasks with high accuracy simply by seeing a few examples or natural-language prompts—no fine-tuning or task-specific gradient updates required. By scaling up model size, the authors demonstrate that a single, general-purpose model can approach or surpass state-of-the-art benchmarks across diverse domains, suggesting a fundamentally new paradigm for building adaptable, versatile AI that learns more like humans and less like traditional supervised systems.

---

## 1. Executive Summary
This paper introduces GPT‑3, a 175‑billion‑parameter transformer language model trained to predict the next token in text, and shows that simply prompting it with natural‑language instructions and a few examples enables strong “in‑context” learning on many tasks—often without any gradient updates. The significance is twofold: (1) scaling model size leads to smooth, predictable gains in few‑shot performance across diverse NLP tasks (Figure 3.1), and (2) at sufficient scale, a single, task‑agnostic model can approach or match fine‑tuned systems on several benchmarks while also demonstrating novel on‑the‑fly reasoning abilities (Sections 3.2–3.9).

## 2. Context and Motivation
- Problem gap
  - Modern NLP relied on “pretrain then fine‑tune”: models are trained on large unlabeled corpora and then fine‑tuned on thousands of labeled examples for each downstream task. This demands task‑specific datasets, risks overfitting to narrow distributions, and doesn’t match how humans often learn from brief instructions or a few examples (Section 1).
- Why it matters
  - Practical impact: many useful tasks (e.g., grammar correction, novel word use) lack large labeled datasets.
  - Scientific impact: if a single model can adapt to new tasks from text alone, we move toward more general and flexible language systems (Figures 1.1–1.3).
- Prior approaches and limits
  - Fine‑tuned SOTA models (e.g., BERT/T5/RoBERTa) achieve strong performance but require per‑task supervised data and retraining (Section 2, Figure 2.1).
  - Earlier “zero-/few‑shot” attempts with smaller LMs showed promise but trailed far behind fine‑tuning (Section 1; e.g., low accuracy on Natural Questions).
- Positioning
  - This work tests whether massive scale alone—without task‑specific gradient updates—yields strong task‑agnostic, few‑shot performance, and whether the “inner loop” of adaptation can happen inside the forward pass via “in‑context learning” (Figure 1.1).

Terminology (defined only when uncommon or paper‑specific):
- `In‑context learning`: specifying a task to the model by writing instructions and a few input‑output examples directly in the prompt; the model adapts within its forward pass (no weight updates).
- `Few‑shot`/`one‑shot`/`zero‑shot`: respectively, many (10–100), one, or no demonstrations in the prompt (Figure 2.1).
- `Closed‑book QA`: answering questions using only knowledge in the model’s parameters (no retrieval).
- `Open‑domain QA`: answering with an external retrieval system over documents.

## 3. Technical Approach
Step‑by‑step overview of how GPT‑3 is built and evaluated:

1) Model architecture (Section 2.1; Table 2.1)
- A transformer language model (decoder‑only) with 8 sizes from 125M to 175B parameters.
- GPT‑3 (175B) uses 96 layers, model dimension 12,288, 96 attention heads, 2048‑token context window, and alternates dense and locally banded sparse attention patterns (to reduce compute while preserving long‑context capacity).
- Training uses model parallelism across depth and width to fit large matrices on GPUs.

2) Training data and filtering (Section 2.2; Table 2.2; Appendix A)
- Base corpus is Common Crawl (2016–2019) filtered for quality using a logistic regression classifier trained to resemble “high‑quality” reference corpora (WebText, Wikipedia, books). Documents are retained with a Pareto‑weighted sampling favoring higher classifier scores.
- Fuzzy deduplication removes near‑duplicates within and across datasets to reduce redundancy and overfitting.
- Final mixture by sampling weight (not proportional to corpus size):
  - 60% filtered Common Crawl (~410B tokens),
  - 22% WebText2 (19B),
  - 8% Books1 (12B),
  - 8% Books2 (55B),
  - 3% Wikipedia (3B).
- Each model trains for 300B tokens total; some datasets are seen <1 epoch, others multiple times (Table 2.2).

3) Training process (Section 2.3; Appendix B; Figure 2.2)
- Optimizer: Adam; cosine LR decay; warmup; gradient clipping; weight decay.
- Batch sizes scale with model size (up to 3.2M tokens for 175B; Table 2.1).
- Total compute grows massively with size; for 175B the estimate is ~3,640 PF‑days (Appendix D; Figure 2.2).

4) Evaluation methodology (Section 2.4; Figure 2.1)
- For each task, construct a prompt with K demonstrations (few‑shot), 1 demonstration (one‑shot), or just instructions (zero‑shot). K is limited by the 2048‑token context (typically 10–100 examples).
- Scoring:
  - Multiple choice: compute the conditional log probability of each candidate answer given the prompt; usually normalize per token. On ARC, OpenBookQA, and RACE, an additional normalization divides by the unconditional probability of the completion to reduce length and frequency bias.
  - Free‑form generation: use beam search (beam=4, length penalty α=0.6; Section 2.4) and score with F1, BLEU, or exact match as standard.
- Example of “programming by prompting”:
  - LAMBADA is recast as a fill‑in‑the‑blank cloze with one‑word targets (Table 3.2; Figure 3.2), letting the model infer the task constraints from the prompt examples.

5) Measuring contamination (Section 4; Appendix C)
- Post‑hoc, identify benchmark overlap with pretraining data using conservative N‑gram matching and re‑evaluate on the “clean” subset to estimate inflation (Figure 4.2).
- Two tasks flagged with small effects: PIQA (−4% relative) and Winograd (−2.6 points; Section 4).

Why this approach?
- The design isolates the role of scale and prompting. By avoiding fine‑tuning and task‑specific architectures, any gains can be attributed to emergent in‑context learning ability and the breadth of the pretrained distribution (Section 2; Figures 1.2, 3.1).

## 4. Key Insights and Innovations
1) Scaling enables effective in‑context learning (fundamental)
- Accuracy increases smoothly with model size and with number of in‑prompt examples (Figures 1.2, 3.1, 3.8). The gap between zero‑, one‑, and few‑shot widens with scale, indicating the larger model exploits demonstrations more effectively.

2) “Prompting as programming” is a universal interface (conceptual + practical)
- Many tasks are solvable by writing instructions and showing a handful of formatted examples—no gradient steps or task‑specific heads (Figure 2.1; Section 2.4). This reframes task specification from “retrain the model” to “author a prompt,” demonstrated from translation (Table 3.4) to cloze (Table 3.2) to arithmetic and word manipulation (Figures 3.10–3.11).

3) Broad, emergent abilities beyond standard benchmarks (novel capability)
- Without task‑specific training, GPT‑3 carries out multi‑digit arithmetic to substantial accuracy, scrambles/unscrambles words, and solves SAT analogies (Section 3.9). For example:
  > “2‑digit addition: 100%; 3‑digit addition: 80.4%; 2‑digit multiplication: 29.2%” (Table 3.9).  
  > “Random‑insertion unscrambling (few‑shot): 67.2%” (Table 3.10).

4) Systematic contamination measurement at scale (methodological)
- The paper builds an explicit clean‑subset evaluation to quantify training/test overlap and its impact (Section 4; Figure 4.2), a practice that becomes essential at web scale. Most benchmarks show negligible differences between full vs. clean subsets; PIQA and Winograd are noted with asterisks due to small effects.

5) Human detectability of model‑generated news drops near chance (societal signal)
- With short (~200 words) or longer (~500 words) articles, human accuracy at distinguishing GPT‑3 outputs is ~52%—barely above random guessing (Tables 3.11, 3.12; Figure 3.13), while weaker models are easier to detect. This highlights both quality and potential misuse risks.

## 5. Experimental Analysis
Evaluation design
- Breadth: >40 datasets across 9 categories (Sections 3.1–3.9), always reporting zero‑, one‑, and few‑shot and scaling across 8 model sizes (Appendix H).
- Metrics: task‑standard (accuracy, F1, BLEU, perplexity); careful scoring details for multiple‑choice and generation (Section 2.4).
- Baselines: compare to fine‑tuned SOTAs (e.g., T5‑11B, RoBERTa) and, for QA, to both closed‑book and retrieval‑augmented systems like RAG (Table 3.3).

Headline quantitative results (few highlights; see cited tables/figures for details)
- Language modeling and cloze/completion (Section 3.1)
  - PTB (zero‑shot perplexity):  
    > “20.5” vs prior zero‑shot 35.8 (Table 3.1).
  - LAMBADA:  
    > “Few‑shot 86.4% accuracy” (Table 3.2, Figure 3.2), +18 points over prior SOTA; zero‑shot 76.2%; one‑shot 72.5%.
  - HellaSwag:  
    > “Few‑shot 79.3%,” exceeding fine‑tuned GPT‑2‑style baselines but below overall fine‑tuned SOTA 85.6% (Table 3.2).
  - StoryCloze:  
    > “Few‑shot 87.7%,” still ~4 points behind fine‑tuned SOTA 91.8% (Table 3.2).

- Closed‑book QA (Section 3.2; Table 3.3; Figure 3.3)
  - TriviaQA:  
    > “Zero‑shot 64.3%, one‑shot 68.0%, few‑shot 71.2%,” which matches or exceeds fine‑tuned open‑domain RAG (68.0%).
  - WebQuestions:  
    > “Few‑shot 41.5%,” approaching fine‑tuned closed‑book T5‑11B+SSM (44.7%).
  - Natural Questions:  
    > “Few‑shot 29.9%,” below fine‑tuned closed‑book T5‑11B+SSM (36.6%); large gains from zero‑shot suggest distribution/style mismatch mitigated by demonstrations.

- Translation (Section 3.3; Table 3.4; Figure 3.4)
  - One/few‑shot performance approaches or surpasses prior unsupervised NMT into English (e.g., Ro→En few‑shot 39.5 BLEU vs mBART 30.5), but from English to Romanian lags (En→Ro 21.0 BLEU few‑shot). Performance scales smoothly with size; translation into English is consistently stronger.

- Winograd‑style coreference (Section 3.4; Table 3.5; Figure 3.5)
  - Winograd (WSC273):  
    > “~88–90% across zero/one/few‑shot,” near human level, but flagged for contamination with a small measured effect (Section 4).
  - WinoGrande (adversarial):  
    > “Few‑shot 77.7%,” below fine‑tuned SOTA 84.6% but competitive with fine‑tuned RoBERTa‑large.

- Commonsense reasoning (Section 3.5; Table 3.6; Figure 3.6)
  - PIQA:  
    > “Few‑shot 82.8% (test‑server),” surpassing fine‑tuned RoBERTa SOTA 79.4% (marked with an asterisk due to small potential contamination).
  - ARC‑Challenge:  
    > “Few‑shot 51.5%,” well below UnifiedQA SOTA 78.5%.
  - OpenBookQA:  
    > “Few‑shot 65.4%,” below SOTA 87.2% but similar to fine‑tuned BERT‑large baselines.

- Reading comprehension (Section 3.6; Table 3.7; Figure 3.7)
  - CoQA:  
    > “Few‑shot 85.0 F1,” close to human/SOTA (~90–91).
  - SQuAD 2.0:  
    > “Few‑shot 69.8 F1,” modest.
  - DROP:  
    > “Few‑shot 36.5 F1,” far below numerical‑reasoning SOTAs (~89).
  - RACE:  
    > “High school 46.8% acc,” comparatively weak.

- SuperGLUE (Section 3.7; Table 3.8; Figure 3.8)
  - Overall few‑shot test score:  
    > “71.8,” competitive with fine‑tuned BERT‑Large (69.0) but below fine‑tuned SOTA (89.0).
  - Strong tasks: COPA (92.0), ReCoRD F1 (91.1); Weak task: WiC (49.4%, at chance).

- NLI (Section 3.8; Figure 3.9; Appendix H)
  - ANLI Round 3 dev:  
    > “Few‑shot 40.2%,” only a modest improvement over chance (33%).

- Synthetic reasoning and pattern tasks (Section 3.9; Tables 3.9–3.10; Figures 3.10–3.12)
  - Arithmetic up to 5 digits shows partial competence; one/few‑shot helps substantially.
  - Word scrambling: robust on several variants in few‑shot (up to ~67%).
  - SAT analogies:  
    > “Few‑shot 65.2%,” above historical human average (57%).

- Human detection of synthetic news (Section 3.9.4; Tables 3.11–3.12; Figure 3.13)
  - Mean human accuracy drops from 86% on a deliberately bad control model to ~52% on GPT‑3, trending toward chance as model size grows.

Robustness, ablations, and caveats
- Scaling law holds: validation loss follows a power law in compute/parameters (Figure 3.1).
- Few‑shot benefits increase with K examples and model size (Figure 3.8).
- Contamination analysis shows small or negligible inflation for most tasks; LAMBADA had large overlap but near‑zero measured effect; PIQA and Winograd show small effects and are marked (Section 4; Figure 4.2).
- No component‑wise ablations (e.g., attention variants) are presented; the study isolates the effect of scaling and prompting rather than architectural tweaks.

Overall assessment
- The experimental evidence strongly supports the central claim: scale makes in‑context learning broadly effective. However, results are mixed: GPT‑3 approaches fine‑tuned SOTA on some tasks (TriviaQA, COPA, ReCoRD, CoQA) but lags on others (DROP, ARC‑Challenge, WiC, RACE). The breadth of tasks and the consistent scaling trends make the case convincing despite these gaps.

## 6. Limitations and Trade-offs
Assumptions and scope
- Assumes task specification fits within a 2048‑token prompt; complex tasks may exceed this limit, constraining K (Section 2.4).
- Uses a unidirectional LM objective; tasks that benefit from bidirectional context (e.g., WiC, some NLI, and span‑based comprehension) may be disadvantaged (Section 5).

Data and contamination
- Web‑scale training risks train–test overlap; despite best efforts, a filtering bug left some overlaps (Section 4). Clean‑subset analysis mitigates but cannot guarantee unbiased estimates.
- Training data are predominantly English (93% by word count), limiting multilingual performance, especially from English into some languages (Section 3.3).

Computation and efficiency
- Training is extremely compute‑intensive (≈3,640 PF‑days for 175B; Appendix D; Figure 2.2). Inference is also heavy and prompts consume context budget; distillation or retrieval could reduce costs (Section 5; 6.3).

Capabilities and failure modes
- Weak on tasks requiring:
  - Fine‑grained sentence comparison or word‑sense discrimination (WiC; Table 3.8).
  - Multi‑step discrete reasoning and numeracy (DROP; Table 3.7).
  - Knowledge‑dense, Wikipedia‑style factual detail in closed‑book QA (NQ; Table 3.3).
- On some synthetic tasks, one‑shot/zero‑shot trails few‑shot, indicating dependence on explicit demonstrations (Tables 3.9–3.10).

Bias and safety
- Bias analyses reveal stereotypical associations by gender, race, and religion (Section 6.2):
  > Occupations more likely to be followed by male identifiers; female co‑occurrences skew toward appearance words like “beautiful/gorgeous” (Table 6.1).  
  > Sentiment trends: “Asian” highest, “Black” lowest across models (Figure 6.1).  
  > For “Islam,” words like “terrorism/violent/terrorist” appear among highly favored co‑occurrences (Table 6.2).
- Human detection experiments show near‑chance detectability of GPT‑3‑generated news (Tables 3.11–3.12), raising misuse concerns (Section 6.1).

Open questions
- Does the model learn new tasks on the fly vs. recognize seen patterns? The paper notes the ambiguity and suggests it likely varies by task (Section 5).
- How much can retrieval, grounding, or RL‑based objectives complement pure next‑token prediction (Section 5)?

## 7. Implications and Future Directions
Shifts in the field
- Validates “prompting as an interface”: instead of building bespoke fine‑tuned models, users can steer a single model with instructions and examples (Figure 2.1). This foreshadows instruction‑following and alignment trends.
- Reinforces scaling laws: predictable gains motivate systematic exploration of larger models—while foregrounding the need for energy‑ and data‑efficient training (Figure 3.1; Section 6.3).

Research avenues
- Architectural and objective augmentations:
  - Bidirectional or encoder‑decoder pretraining combined with in‑context learning to improve comparison/entailment tasks (Section 5).
  - Retrieval‑augmented pretraining/inference (as contrasted with RAG in Table 3.3) to boost knowledge‑heavy QA.
  - Multi‑modal grounding (vision, action) to improve “commonsense physics” and world modeling (Section 5).
  - Learning objectives that prioritize entities/relations or goal‑directed behavior beyond next‑token prediction (Section 5).
- Prompt engineering and tooling:
  - Methods to optimize prompts automatically and exploit demonstration order and formatting.
  - Extended context windows and memory mechanisms to admit larger K or longer documents.
- Efficiency and safety:
  - Distillation to reduce inference cost while retaining few‑shot abilities (Section 5).
  - Robust, standardized contamination detection; benchmark design less likely to appear verbatim on the web (Section 4).
  - Systematic bias auditing and mitigation integrated into pretraining and prompting (Section 6.2).
  - Detection and provenance tools for synthetic text (Section 3.9.4).

Applications
- High‑leverage, low‑label settings: rapid prototyping of classifiers, content transformation (summarization, style transfer), data augmentation, code and writing assistance, conversational agents.
- Knowledge tasks where closed‑book few‑shot is sufficient (e.g., TriviaQA‑like) and creative generation where prompts can steer style and content (Sections 3.2, 3.9.4, F).

> Bottom line: The paper demonstrates that sheer scale, coupled with carefully designed prompts, yields a single model that can perform a wide array of tasks without task‑specific training. The approach doesn’t replace fine‑tuning everywhere yet, but it decisively establishes in‑context learning as a core capability and a practical interface for general‑purpose language models.
