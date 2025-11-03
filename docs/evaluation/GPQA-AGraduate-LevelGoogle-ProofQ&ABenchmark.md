# GPQA: A Graduate-Level Google-Proof Q&A Benchmark

**ArXiv:** [2311.12022](https://arxiv.org/abs/2311.12022)

## 🎯 Pitch

GPQA introduces a rigorous, graduate-level benchmark of 448 expert-written questions in biology, physics, and chemistry that are designed to be 'Google-proof': even highly skilled non-experts with full web access struggle to answer them, while top-tier models like GPT-4 also perform poorly compared to domain experts. By providing a testbed where only subject-matter experts can reliably discern the correct answers, GPQA enables realistic, high-stakes research on scalable oversight—how to reliably supervise and align frontier AI systems when even well-resourced human evaluators lack the expertise to verify answers themselves. This dataset directly targets the critical challenge of ensuring AI safety and truthfulness as models advance into domains beyond most human supervisors' reach.

---

## 1. Executive Summary
GPQA introduces a small but unusually hard, expert‑written benchmark of 448 multiple‑choice questions in biology, physics, and chemistry designed to be “Google‑proof.” Its core contribution is a data creation and validation pipeline that yields questions with high expert agreement yet low accuracy for skilled non‑experts and current frontier models, enabling realistic experiments on scalable oversight—how to supervise AI systems on tasks that even well‑resourced non‑experts cannot solve or verify.

## 2. Context and Motivation
- Problem addressed
  - Scalable oversight: how to reliably supervise models on questions whose truth a supervisor cannot easily produce or verify. This setting becomes critical as models edge into superhuman or specialist territory. Definition: “scalable oversight” (Section 1) refers to eliciting and checking truthful answers from models even when supervisors lack the knowledge or time to compute or confirm the truth themselves.
  - Existing QA benchmarks are usually solvable by non‑experts using web search or are written by non‑experts; they do not stress the supervision challenge near or beyond the frontier of human expertise (Section 5).

- Why this matters
  - If future models help generate new scientific knowledge, supervisors must detect errors, hallucinations, and sycophancy without having the answers in hand (Section 1). Oversight methods like RLHF depend on the supervisor’s ability to judge outputs; when the task exceeds the supervisor’s expertise, these methods can fail.

- Prior approaches and shortcomings
  - Crowdsourced or curated QA datasets (e.g., SQuAD, Natural Questions, TriviaQA, MMLU) rely on information that non‑experts can often locate or reason through with short references (Section 5). They do not ensure that:
    - The true answer is difficult for non‑experts even with internet access.
    - Incorrect options are plausible to skilled non‑experts.
  - QuALITY targets long‑context reading difficulty rather than real subject‑matter expertise (Section 5).

- Positioning
  - GPQA constructs questions written and validated by PhD‑level domain experts, and it explicitly verifies that skilled non‑experts (experts in other domains) fail even with unrestricted web access and ample time (Sections 2.1, 3.2). It thus provides a testbed suited to research on scalable oversight methods like debate, market‑making, and recursive reward modeling (Sections 1, 2.1).

## 3. Technical Approach
GPQA’s methodology is a four‑stage expert‑centered pipeline that jointly enforces objectivity (clear, correct answer) and difficulty (non‑experts cannot solve even with the web).

- Participants and qualification (Section 2.1)
  - 61 Upwork contractors who have completed or are pursuing PhDs in their field and are proficient in English; high‑rated freelancers preferred.
  - Writers and validators are experts in one of three high‑level domains (biology, physics, chemistry) and multiple subdomains (Section 2.2).
  - Non‑expert validators are experts in other domains (e.g., a physicist validates biology or chemistry), with unlimited web access but no LLMs; they must spend at least 15 minutes and often far more (median 30 min; mean 37 min; top 20% ≥45 min; Section 3.2).

- Stage 1 — Question writing (Section 2.1; Appendix A.5.1)
  - Experts write hard, stand‑alone, four‑choice questions designed to be answerable by in‑domain experts even without seeing options.
  - Writers supply detailed explanations: why the correct choice is correct and why each distractor is plausible but wrong. These explanations later seed few‑shot prompts for model baselines and aid in objectivity checks.
  - Writers are explicitly instructed to avoid “easy tells” (surface patterns that leak the answer), to minimize rote calculation, and to consider “search resistance” (run the web searches a non‑expert would try and ensure they don’t trivially solve the question).

- Stage 2 — First expert validation and feedback (Section 2.1; Appendix A.5.2)
  - Another expert in the same domain answers the question and provides detailed feedback aimed at improving clarity, objectivity, and difficulty.
  - Writers then revise the question (Stage 3) based on this feedback (Appendix A.5.3).

- Stage 3 — Question revision (Section 2.1)
  - Writers may revise wording, assumptions, or distractor plausibility to fix ambiguities or strengthen difficulty, while retaining a unique correct answer.

- Stage 4 — Second expert validation and non‑expert validation (Section 2.1)
  - A second expert answers and comments on the revised question; three non‑experts (experts in other fields) attempt the question with full web access (no LLMs).
  - Both experts also report whether they have sufficient expertise to answer; this supports objectivity analysis (Section 3.1).

- Incentive design to reinforce objectivity and difficulty (Section 2.1; Appendix A.4)
  - Writers: $10 base + $20 per expert who answers correctly + $15 per non‑expert who answers incorrectly + $30 bonus for questions where both experts are correct and ≥2/3 non‑experts are wrong.
  - Expert validators: $10 base + $10 if they answer correctly; the first validator also gets bonuses if their feedback improves objectivity/difficulty (e.g., if the second expert is correct and most non‑experts are wrong). A flat $7 is granted to account for role asymmetry.
  - Non‑experts: $10 base + $30 for answering correctly—ensuring they really try.

- Measuring and filtering for objectivity (Section 3.1)
  - “Post‑hoc agreement”: After seeing the writer’s explanation, experts state whether they now agree the answer is uncontroversially correct and explain mistakes if any.
  - Manual analysis of 191 second‑expert errors classifies feedback into eight categories (Table 4), separating clear validator mistakes from true question issues (ambiguity, missing assumptions, or wrong answers). These judgments control which questions enter the main and “Diamond” splits.

- Data splits (Section 2.3)
  - Extended: 546 questions (18 held out, unreleased).
  - Main (GPQA): 448 questions; filters out clearly non‑objective/easy items and includes cases where an expert erred but later demonstrated understanding of their mistake.
  - Diamond: 198 hardest, highest‑confidence questions; requires stronger expert agreement (both experts correct, or one expert’s mistake is clearly explained) and a majority of non‑experts wrong.

- Domains and coverage (Section 2.2; Table 3)
  - Biology (Molecular Biology; Genetics), Physics (nine subdomains including Quantum Mechanics, HEP, Astrophysics), Chemistry (notably many in Organic Chemistry). Chemistry questions are especially hard for non‑experts while maintaining high expert accuracy (Table 3).

- Quality checks against shortcuts (Appendix A.2)
  - Answer‑only classifiers (fine‑tuned T5 and a RoBERTa‑based CBOW) trained to guess the correct option using only the text of the choices and randomized order perform at chance (≈25%), suggesting no easy lexical tells in the options.

- Baselines and open‑book tool use (Section 4; Appendix A.3)
  - Closed‑book with four prompting styles: zero‑shot, few‑shot, zero‑shot chain‑of‑thought (CoT), and few‑shot CoT.
  - Open‑book: GPT‑4 with a “self‑ask” style tool‑use prompt that iterates between sub‑questions and web searches; high abstention is mitigated with a backoff to few‑shot CoT when the search run abstains.

Note on examples: The paper includes sample questions (Table 1, Figure 1), but it asks readers not to reproduce dataset items online to avoid training‑data leakage; this analysis does not restate them.

## 4. Key Insights and Innovations
- A pipeline that simultaneously enforces objectivity and difficulty
  - Novelty: Two expert validations with post‑hoc agreement plus explicit non‑expert testing under unrestricted web access (no LLMs) and ample time (avg 37 minutes) (Sections 2.1, 3.2). This goes beyond prior QA benchmarks that rarely verify both conditions together.
  - Significance: Creates a realistic setting where a non‑expert supervisor cannot simply Google the answer—crucial for testing scalable oversight protocols.

- Incentive structure tuned for data quality, not speed
  - Different from typical crowdsourcing: payments are dominated by quality‑based bonuses aligning all roles toward objective, hard questions (Appendix A.4). This helps avoid easy or ambiguous items and encourages careful reasoning and feedback.

- Evidence‑based objectivity measurement
  - Beyond “expert accuracy”: post‑hoc agreement and manual error categorization (Table 4) separate validator mistakes from flawed questions (Section 3.1). With these filters, estimated objectivity rises to roughly three‑quarters of questions on the Extended set (≈74–76%; Section 3.1).

- “Google‑proof” difficulty validated quantitatively
  - Skilled non‑experts with web access score ≈34% (Extended), far above chance but far below experts (≈65%) and with substantial time investment (Figure 2; Section 3.2). This demonstrates real supervision difficulty rather than trick questions or time pressure.

- Tool‑use baseline that stresses current limitations
  - GPT‑4 with web search shows only marginal gains over few‑shot CoT and high abstention (Table 5; Table 7), highlighting that even strong models struggle to operationalize tool‑assisted research workflows on these expert questions (Section 4).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and splits: Extended (546), Main (448), Diamond (198) (Section 2.3). The Main and Diamond sets are filtered to prefer expert agreement and non‑expert difficulty.
  - Participants: Two experts per question (first pre‑revision, second post‑revision); three non‑experts per question with full web access (no LLMs) (Section 2.1).
  - Metrics: Multiple‑choice accuracy; abstention rate for models; expert/non‑expert self‑reported expertise and confidence; difficulty ratings (Sections 3.1–3.2; Appendix A.2).
  - Baselines: Llama‑2‑70B‑chat, GPT‑3.5‑turbo‑16k, GPT‑4; closed‑book (four prompts) and open‑book GPT‑4 with search (Section 4; Table 6).

- Core quantitative results
  - Human accuracies (Table 2):
    > Extended: Experts 64.8%, Non‑experts 34.1%, with 90.7% of expert validators reporting “sufficient expertise.”  
    > Main: Experts 71.9%*, Non‑experts 30.4%*.  
    > Diamond: Experts 81.3%*, Non‑experts 22.1%*.  
    Asterisks denote selection bias from filtering.
  - Expertise gap by domain (Table 3, Extended):
    > Biology: 66.7% (experts) vs 43.2% (non‑experts)  
    > Physics: 57.3% vs 32.5%  
    > Chemistry: 72.0% vs 31.4% (largest gap: +40.6 points)
  - Expert objectivity and mistakes (Section 3.1; Table 4):
    > Post‑hoc agreement: ≈80% (first expert), ≈85% (second expert) on Extended.  
    > Manual analysis of 191 second‑expert errors: 46% clearly acknowledge mistake or demonstrate understanding of the writer’s reasoning (categories 1–2 in Table 4).  
    > Estimated objectivity after excluding such validator mistakes: ≈73.6–76.4%.
  - Non‑expert effort and difficulty correlation (Section 3.2; Figure 3):
    > Non‑experts spend median 30 minutes; mean 37 minutes.  
    > Higher expert‑rated difficulty correlates with lower non‑expert accuracy (Figure 3).
  - Model baselines (Tables 5–7; Section 4):
    - Closed‑book few‑shot CoT:
      > Llama‑2‑70B: 30.4% (Extended)  
      > GPT‑3.5‑turbo‑16k: 28.2%  
      > GPT‑4: 38.7% (Extended), ≈39–40% on Main/Diamond
    - Open‑book GPT‑4 with web search:
      > 39.4% (Extended), 41.0% (Main), 38.8% (Diamond) but with 37.2% abstention on Main (Table 7). The paper uses a backoff to few‑shot CoT when the search run abstains (Section 4).
    - GPT‑4 domain breakdown (few‑shot CoT; Section 4): biology ≈58.1%, physics ≈37.0%, chemistry ≈31.8%.
  - Calibration (Appendix A.2; Figure 5):
    > Expected Calibration Error (ECE): experts 0.1259, non‑experts 0.1188, but non‑experts are “extremely overconfident” in most bins except 25%; the heavy mass at 25% lowers their aggregate ECE.
  - Anti‑shortcut checks (Appendix A.2):
    > Answer‑only models perform at chance (~25%) on a 20% validation split, suggesting no major exploitable artifacts in option text.

- Do the experiments support the claims?
  - Yes, for both objectivity and difficulty:
    - Objectivity: two‑expert validation with post‑hoc agreement and manual categorization demonstrates most disagreements are validator mistakes rather than flawed questions (Section 3.1; Table 4).
    - Difficulty: skilled non‑experts, with time and web access, achieve only ≈34% (Extended) and even lower on filtered splits (Table 2), matching the “Google‑proof” goal (Section 3.2; Figure 2).
    - Models: GPT‑4 remains far from expert level and near or slightly above non‑experts, confirming headroom for oversight research (Table 5).

- Ablations, failure cases, robustness
  - Filtering to Main/Diamond reduces non‑expert accuracy but model accuracy does not consistently differ across splits (selection effects make non‑expert numbers biased downward; Section 4).
  - GPT‑4 with search exhibits high abstention (≈37–41%; Table 7), implying difficulty in orchestrating tool use on these questions (Section 4).
  - Anti‑artifact ablation (answer‑only) and calibration analyses (Appendix A.2) add robustness: no obvious option‑text shortcuts; non‑experts are overconfident—a realistic oversight pitfall.

- Caveats in interpretation
  - Selection bias in Main and Diamond: accuracies on those sets are not directly comparable to Extended (Tables 2 and 5).
  - The retrieval prompt and tool integration may not be optimized; so open‑book results are indicative, not upper bounds (Section 4).

## 6. Limitations and Trade-offs
- Dataset size and statistical power (Section 6)
  - Main split has 448 items; Diamond 198. This is adequate for evaluations and oversight experiments but not ideal for training or for detecting small accuracy deltas.

- Specialized non‑experts and ecological validity (Section 6)
  - Non‑experts are themselves highly skilled specialists in other fields; their performance may overestimate what typical annotators could achieve and may not reflect all real‑world oversight settings.

- Domain and annotator biases (Section 6)
  - Sourced from Upwork without demographic/region balancing; topics and phrasing may reflect the contributors’ backgrounds. The dataset is not claimed to be representative of scientific practice broadly.

- Applicability to truly superhuman settings (Section 6)
  - GPQA approximates, but does not equal, supervising superhuman systems on questions with unknown ground truth. The paper suggests a future direction: assemble hard unanswered questions that later receive definitive answers.

- Assumptions and exclusions
  - Non‑experts have web but no LLMs; this isolates the oversight challenge but differs from future workflows where non‑experts may use strong tools.
  - Questions are multiple‑choice; while writers aim for stand‑alone solvability without options, the released format is MCQ.

- Computational/tooling constraints in baselines (Section 4)
  - The open‑book baseline uses a single “self‑ask” design with high abstention; stronger tool‑use scaffolds could yield different results.

## 7. Implications and Future Directions
- How this changes the landscape
  - GPQA provides a rare, vetted benchmark where:
    - True answers are known and vetted by experts with post‑hoc adjudication (Section 3.1).
    - Non‑experts cannot solve the problems even with the internet and abundant time (Section 3.2).
    - Frontier models perform well below experts (Tables 5–6).
  - This combination uniquely supports research on scalable oversight protocols (debate, market‑making, recursive reward modeling) by making it non‑trivial for supervisors to judge correctness without the model’s help (Sections 1–2).

- Enabled research directions
  - Protocol design and testing:
    - Compare supervision methods on GPQA (e.g., debate vs. single‑assistant explanations) and measure non‑expert uplift toward expert accuracy.
    - Study sycophancy and hallucination under supervision pressure (Section 1 references).
  - Tool‑use and retrieval:
    - Develop better scaffolds for search, reading, and citation that reduce abstention and increase accuracy (Table 7 suggests current designs underperform).
  - Data creation for superhuman evaluation:
    - Pilot the paper’s suggestion to curate questions currently unanswered but likely to be resolved soon, to test whether oversight methods can find correct answers ahead of consensus (Section 6).

- Practical applications
  - Training and evaluating human supervisors: calibrate how much assistance and what protocol structure (e.g., argumentation, evidence presentation) is needed for non‑experts to accurately evaluate specialist answers.
  - Model governance: track progress of models on tasks that resist naive web search, informing deployment thresholds for high‑stakes scientific assistance.

- Extensions to the dataset
  - Increase size and domain diversity (engineering, law, medicine) while retaining expert vetting and non‑expert difficulty.
  - Add free‑response variants and rubric‑based grading to evaluate generative reasoning beyond multiple choice.
  - Release richer metadata (e.g., expert difficulty ratings, references) already collected to facilitate nuanced analyses (Figures 3–5; Appendix A.2).

> Key result snapshot (Tables 2, 5, 7):  
> • Experts: ≈65% on Extended; ≈81% on Diamond (selection‑biased upward).  
> • Skilled non‑experts with full web: ≈34% on Extended; ≈22% on Diamond.  
> • Best model (GPT‑4 few‑shot CoT/open‑book with backoff): ≈39–41%; high abstention in open‑book (≈37–41%).  
> These numbers demonstrate substantial headroom for oversight research and tool‑use improvements.

Definitions used
- `Google-proof`: difficult for non‑experts even with unrestricted web search and significant time.
- `Scalable oversight`: supervising models on tasks where supervisors cannot easily compute or verify the truth; methods aim to extract correct answers reliably.
- `Chain‑of‑thought (CoT) prompting`: prompting a model to write step‑by‑step reasoning before finalizing an answer.
- `Self‑ask` tool use: an iterative procedure where the model decomposes a question into sub‑questions and issues web searches to answer them sequentially.
- `Post‑hoc agreement`: after seeing the ground‑truth explanation, an expert validator indicates whether the answer is uncontroversially correct and, if they erred, explains their mistake.
- `Abstention rate`: fraction of questions for which the model declines to provide a final choice.
