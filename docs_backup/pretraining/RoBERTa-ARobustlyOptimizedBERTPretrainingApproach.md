# RoBERTa: A Robustly Optimized BERT Pretraining Approach

**ArXiv:** [1907.11692](https://arxiv.org/abs/1907.11692)

## 🎯 Pitch

RoBERTa uncovers that substantial improvements in BERT-style language models come not from architectural changes, but from optimizing pretraining strategies—removing next sentence prediction, leveraging much more data, using larger batches, dynamic masking, and longer training. By rigorously dissecting and enhancing BERT’s training recipe, RoBERTa delivers state-of-the-art results on GLUE, SQuAD, and RACE, demonstrating that careful attention to data and optimization outpaces recent model innovations and setting a new standard for future NLP research and applications.

---

## 1. Executive Summary
RoBERTa (“Robustly optimized BERT approach”) reexamines how to pretrain BERT-style language models and shows that careful choices about data, batch size, sequence length, masking, and training duration—not a new model architecture—deliver large gains. By removing the next sentence prediction objective, using dynamic masking, training on much more data with larger batches and longer schedules, and modestly changing tokenization, RoBERTa achieves state-of-the-art results on GLUE, SQuAD, and RACE (Sections 4–5; Tables 4–7).

## 2. Context and Motivation
- Problem/gap addressed
  - Recent pretraining methods (ELMo, GPT, BERT, XLM, XLNet) report impressive results, but comparisons are muddied by different datasets, training budgets, and hyperparameters. Training is computationally expensive, so many design choices were underexplored (Section 1).
  - The paper asks: which parts of BERT’s pretraining recipe matter most (data size, objectives, sequence length, masking strategy, batch size), and can a better-tuned recipe match or exceed newer methods?

- Importance
  - Practical: Clear guidance on pretraining recipes reduces wasted compute and standardizes comparisons across future models.
  - Scientific: It disentangles objective/architecture innovations from confounding factors like data scale and training time.

- Prior approaches and shortcomings
  - BERT uses masked language modeling (MLM) plus next sentence prediction (NSP) on BookCorpus+Wikipedia (16GB), trained for 1M steps with 256-sequence batches (Sections 2.3–2.5).
  - Later models (e.g., XLNet) changed objectives and trained on substantially more data and larger batches, making it unclear whether gains came from objectives or from scale and training choices (Section 5 intro; Table 4 context).

- Positioning
  - This work replicates BERT in FAIRSEQ, systematically ablates training choices, introduces a larger curated dataset mix (adds CC-NEWS, OpenWebText, Stories; Section 3.2), and aggregates the best choices into `RoBERTa`—same architecture and MLM objective as BERTLARGE, but a stronger pretraining procedure (Sections 4–5).

## 3. Technical Approach
RoBERTa keeps BERT’s architecture and MLM objective but changes how pretraining is done. Key ingredients and how they work:

- Pretraining objective (unchanged from BERT; Section 2.3)
  - `Masked Language Modeling (MLM)`: randomly select 15% of input tokens; of these, replace 80% with `[MASK]`, keep 10% unchanged, and replace 10% with a random token. The model predicts the original tokens using a cross-entropy loss.
  - `Next Sentence Prediction (NSP)`: a binary classifier to predict whether two segments are next to each other in the original text. RoBERTa removes this objective after analysis (Section 4.2).

- Masking strategy: static vs dynamic (Section 4.1)
  - `Static masking`: prepare one masked version of each sentence offline; to get some variety, the dataset is duplicated 10× so each sentence appears with 10 masks over 40 epochs, still repeating masks (Section 4.1).
  - `Dynamic masking`: sample a fresh mask pattern every time a sequence is fed to the model. This scales naturally to longer training or larger datasets without precomputing many duplicates.

- Input formatting variants and the fate of NSP (Section 4.2)
  - BERT’s original `SEGMENT-PAIR+NSP`: concatenate two segments (each can contain multiple sentences) and predict if the second follows the first; limit total length to 512 tokens.
  - `SENTENCE-PAIR+NSP`: same as above but each segment is a single sentence; increases batch size to keep token count comparable.
  - `FULL-SENTENCES` (no NSP): pack full sentences contiguously up to 512 tokens; may cross document boundaries; add an extra separator when crossing.
  - `DOC-SENTENCES` (no NSP): like FULL-SENTENCES but never cross document boundaries; batch size is dynamically increased when reaching short end-of-document sequences to keep token throughput similar.
  - Why remove NSP? Experiments show better downstream performance without it when using the “packed sentences” formats that preserve longer contexts (Table 2; Section 4.2).

- Large-batch training (Section 4.3)
  - Idea: increase batch size substantially (2K–8K sequences) and tune the peak learning rate to keep optimization stable. This improves both training efficiency (parallelism) and model quality.
  - The study controls for total data passes: e.g., 256×1M steps ≈ 2K×125K steps ≈ 8K×31K steps in computational cost (Table 3).

- Tokenization change: byte-level BPE (Section 4.4)
  - Replace BERT’s 30K character-level BPE with a 50K `byte-level BPE` (as in GPT-2), which builds subword units from bytes rather than Unicode characters. Benefits: universal text coverage without “unknown” tokens and no heuristic pre-tokenization. It adds ~15–20M parameters but shows only slight differences in early tests; the universality motivates its use (Section 4.4).

- Data pipeline (Section 3.2)
  - Original BERT data: BookCorpus + English Wikipedia (16GB).
  - Added corpora totaling ~160GB uncompressed:
    - `CC-NEWS` (76GB after filtering): 63M English news articles (Sept 2016–Feb 2019). Collected with `news-please` (footnote 4).
    - `OpenWebText` (38GB): open-source recreation of WebText from Reddit links (footnote 5).
    - `STORIES` (31GB): CommonCrawl subset filtered to “story-like” text (Trinh & Le 2018).
  - The work notes it conflates data size and diversity; disentangling them is left for future work (footnote 9).

- Implementation and hyperparameters (Sections 3.1, 2.4; Appendix Tables 9–10)
  - Framework: FAIRSEQ with mixed-precision training on DGX-1 nodes (8×32GB V100 per node; Section 3.1).
  - Optimizer: Adam with β1=0.9, β2=0.98 (tuned from 0.999 for stability with large batches), ε=1e-6; weight decay 0.01; linear LR warmup then decay (Sections 2.4, 3.1).
  - Sequence length: always train with full 512-token sequences (unlike BERT which used shorter sequences for most steps; Section 3.1).
  - RoBERTaLARGE pretraining schedule (Appendix Table 9): batch size 8K sequences, max 500K steps, peak LR 4e-4, warmup 30K steps; dropout 0.1; GELU activations.
  - Hardware scale: e.g., 100K-step run over Books+Wiki used 1024 V100 GPUs for ~1 day (Section 5 Results paragraph).

- Fine-tuning setups (Sections 3.3, 5.1–5.3; Appendix Table 10)
  - GLUE: single-task fine-tuning with small grids over batch size {16, 32} and LR {1e-5, 2e-5, 3e-5}; early stopping; dev results are medians over 5 seeds (Section 5.1).
  - For leaderboard test, RoBERTa ensembles 5–7 single-task models; uses two task-specific tweaks:
    - `QNLI`: pairwise ranking formulation adopted by many top entries (Section 5.1 “Task-specific modifications”).
    - `WNLI`: reformatted SuperGLUE version; margin ranking loss with noun-phrase candidates extracted by spaCy; uses only positive training examples (Section 5.1).
  - SQuAD: no data augmentation; standard span extraction for v1.1; for v2.0, add a binary “answerable” classifier trained jointly (Section 5.2).
  - RACE: encode each of the 4 candidate-answer concatenations with the passage and question; classify using the [CLS] representation (Section 5.3).

## 4. Key Insights and Innovations
- Removing `NSP` and changing input packing improves performance (Section 4.2)
  - What’s new: Instead of segment pairs with NSP, RoBERTa packs contiguous sentences into full 512-token blocks and drops NSP.
  - Why it matters: Without NSP, both `FULL-SENTENCES` and `DOC-SENTENCES` outperform BERTBASE’s published results. For example, `DOC-SENTENCES` yields SQuAD 1.1/2.0 F1 90.6/79.7, MNLI-m 84.7, SST-2 92.7, RACE 65.6, surpassing the `SEGMENT-PAIR+NSP` format (Table 2). This demonstrates NSP is not required and may hinder learning long-range dependencies in practice.

- Longer training on more data with larger batches is the dominant driver of gains (Section 5; Table 4)
  - What’s new: Scale batch size to 8K sequences and extend training from 100K to 500K steps while adding ~10× more text than BERT.
  - Why it matters: With just Books+Wiki and the improved recipe, RoBERTaLARGE jumps to MNLI 89.0 and SQuAD2.0 F1 87.3 (Table 4), already beating BERTLARGE (MNLI 86.6; SQuAD2.0 F1 81.8).
  - Adding the extra 144GB data and training to 500K steps further improves to MNLI 90.2 and SQuAD2.0 F1 89.4 (Table 4), matching or surpassing XLNetLARGE while using the simpler MLM objective.

- Dynamic masking is at least as good and more scalable than static masking (Section 4.1; Table 1)
  - What’s new: Generate a new mask per presentation instead of precomputing 10 duplicates.
  - Why it matters: Comparable or slightly better dev results (e.g., SQuAD2.0 F1 78.7 vs 78.3; SST-2 92.9 vs 92.5; Table 1) and avoids repeated mask patterns when training longer or on bigger corpora.

- Very large batch sizes can improve optimization and outcomes when LR is tuned (Section 4.3; Table 3)
  - What’s new: Scale batch to 2K–8K sequences and adjust peak LR (7e-4 to 1e-3).
  - Why it matters: Better MLM perplexity (3.99 → 3.68) and small accuracy gains on MNLI (84.7 → 85.2) while keeping total compute constant (Table 3). Large batches also unlock efficient multi-GPU scaling.

- Practical tokenization tweak: byte-level BPE (Section 4.4)
  - What’s new: Switch to GPT-2 style byte-level BPE with a 50K vocab to eliminate unknown tokens and heuristic tokenization.
  - Why it matters: Early tests show only minor performance differences, but the universality and simplicity are operational advantages; RoBERTa adopts it for the main experiments.

Overall, most advances are careful engineering and scaling decisions rather than a new objective or architecture—an important rebalancing of where gains actually come from.

## 5. Experimental Analysis
- Evaluation methodology (Section 3.3)
  - Benchmarks: GLUE (9 tasks; classification and pairwise similarity), SQuAD v1.1 and v2.0 (extractive QA), RACE (multi-choice reading comprehension).
  - Metrics:
    - GLUE: task-specific metrics like accuracy (MNLI, QNLI, SST-2, RTE), F1/accuracy for MRPC/QQP, Matthews corr. for CoLA, Pearson/Spearman for STS-B.
    - SQuAD: EM (Exact Match) and F1.
    - RACE: accuracy.
  - Setup: Fine-tune single-task models; report median dev scores over 5 seeds. For test leaderboards, use ensembling, and for QNLI/WNLI, specific formulations (Section 5.1).

- Ablations that support each design choice
  - Dynamic vs static masking (Table 1)
    > SQuAD2.0 F1: 78.7 (dynamic) vs 78.3 (static); SST-2: 92.9 vs 92.5; MNLI-m identical or within noise (84.0–84.3).
  - Input format and NSP (Table 2)
    > Without NSP, `DOC-SENTENCES` achieves SQuAD2.0 F1 79.7, MNLI-m 84.7, RACE 65.6, all higher than `SEGMENT-PAIR+NSP` (SQuAD2.0 F1 78.7, MNLI-m 84.0, RACE 64.2).  
    > Using single sentences with NSP hurts: `SENTENCE-PAIR+NSP` scores drop across tasks (e.g., SQuAD1.1 F1 88.7 vs 90.4; MNLI-m 82.9 vs 84.0).
  - Large batches (Table 3)
    > With controlled compute, batch 2K reaches MLM perplexity 3.68 (better than 3.99 with batch 256) and MNLI 85.2; batch 8K also reduces perplexity to 3.77 with comparable dev accuracy.
  - Data scale and training length (Table 4)
    > With Books+Wiki only (16GB) and 100K steps: SQuAD1.1/2.0 F1 93.6/87.3, MNLI 89.0.  
    > Add 144GB more data (total ~160GB), 100K steps: SQuAD1.1/2.0 94.0/87.7, MNLI 89.3.  
    > Train longer to 300K steps: 94.4/88.7, MNLI 90.0.  
    > Train to 500K steps: 94.6/89.4, MNLI 90.2; SST-2 96.4.  
    > For reference, BERTLARGE on Books+Wiki is SQuAD1.1/2.0 90.9/81.8, MNLI 86.6 (Table 4).

- Main quantitative results
  - GLUE single-task dev (Table 5, top)
    > RoBERTa achieves best dev results on all 9 tasks; e.g., MNLI 90.2/90.2, QNLI 94.7, SST-2 96.4, CoLA 68.0, STS-B 92.4.  
    > This surpasses BERTLARGE (e.g., MNLI 86.6, SST-2 93.2) and XLNetLARGE (MNLI 89.8, SST-2 95.6).
  - GLUE test leaderboard (Table 5, bottom)
    > RoBERTa ensemble average 88.5, slightly above XLNet’s 88.4. It is SOTA on 4/9 tasks (MNLI, QNLI, RTE, STS-B). Note: test submission uses ranking for QNLI and a special WNLI formulation (Section 5.1).
  - SQuAD (Table 6)
    > Dev: v1.1 F1/EM 94.6/88.9; v2.0 89.4/86.5. Matches or exceeds XLNet on dev without any extra QA data or layerwise LR schedules.  
    > Test (single model): v2.0 89.8/86.8, competitive with top systems using extra data.
  - RACE (Table 7)
    > Test accuracy 83.2% (Middle 86.5, High 81.3), better than XLNetLARGE (81.7).

- Do the experiments support the claims?
  - Yes, the paper includes targeted ablations isolating masking strategy (Table 1), input format/NSP (Table 2), batch size (Table 3), and data/training length (Table 4), then validates the aggregated recipe on major benchmarks (Tables 5–7).
  - Caveats:
    - For GLUE test, RoBERTa uses task-specific formulations (ranking for QNLI; margin ranking for WNLI) and ensembles (Section 5.1), so dev-set single-model comparisons are the fairest head-to-heads with BERT/XLNet.
    - The increase in data size and data diversity is conflated (footnote 9).
    - The paper notes other methods might also improve with more tuning (footnote 2).

- Robustness and failure cases
  - The study emphasizes stability tuning (e.g., Adam ε and β2=0.98 with large batches; Section 3.1), and shows dynamic masking prevents repeated patterns in long training.
  - No explicit failure case analysis is reported, but results suggest gains continue up to 500K steps without overfitting (Section 5, end).

## 6. Limitations and Trade-offs
- Compute intensity
  - Training at RoBERTa’s scale requires substantial hardware (e.g., 1024 V100 GPUs for ~1 day just for a 100K-step run; Section 5). Batch size 8K and 500K steps are beyond many labs’ budgets.

- Data considerations
  - English-only corpora; generalization to other languages is not tested.  
  - Size vs. diversity effects are not isolated (footnote 9). The exact contribution of each added corpus (CC-NEWS, OpenWebText, Stories) is not ablated individually.

- Objective and architecture scope
  - The work keeps BERT’s architecture and MLM objective; it does not explore whether similar scaling and recipe tweaks would further improve alternative objectives (e.g., XLNet) beyond the brief comparison (Table 4).

- Tokenization choice
  - Byte-level BPE is adopted mainly for universality; the paper notes early experiments show slight performance differences but does not deliver a deep analysis (Section 4.4).

- Evaluation comparability
  - Leaderboard results use task-specific modifications for QNLI and WNLI and model ensembling (Section 5.1), which complicates direct, apples-to-apples comparisons to systems without such tweaks.

- NSP interpretation
  - While removing NSP helps in this setup, the study does not rule out that other formulations of inter-sentence objectives or different input formats could be beneficial in other tasks.

## 7. Implications and Future Directions
- How this changes the field
  - It re-centers the narrative: much of the perceived progress after BERT stems from training procedure and scale rather than fundamentally new pretraining objectives. The MLM objective remains highly competitive when trained properly (Abstract; Section 5; Table 4).

- Practical guidance for practitioners
  - To pretrain BERT-like models effectively:
    - Remove `NSP`; pack long contiguous text (`FULL-SENTENCES` or `DOC-SENTENCES`; Section 4.2).
    - Use `dynamic masking` (Section 4.1).
    - Train with `large batches` and tune the learning rate; consider β2=0.98 for Adam (Sections 3.1, 4.3).
    - Prefer full 512-token sequences during pretraining (Section 3.1).
    - Scale data substantially; news, web, and story-like corpora add value (Section 3.2).
    - Byte-level BPE is a robust, simple tokenization choice (Section 4.4).
  - For fine-tuning: single-task fine-tuning suffices for strong dev performance; task-specific formulations can help on leaderboards (Section 5.1).

- Research directions enabled/suggested
  - Systematic disentangling of data size vs. data diversity effects (footnote 9).
  - Reassessment of inter-sentence objectives: when and how should models learn discourse relationships if NSP is removed?
  - Extending the recipe to multilingual settings and other modalities.
  - Sample efficiency: can we get similar gains with fewer steps via better curricula, optimization, or data selection?
  - Understanding why large batches help MLM beyond optimization speed (links to generalization and implicit regularization).

- Applications
  - Any downstream NLP task that benefits from strong language representations: classification, QA, reading comprehension, NLI, and beyond. The released code and models (Abstract; Conclusion) make it straightforward to adopt this recipe in practice.

In short, RoBERTa demonstrates that careful training at scale—longer, larger, and with better input packing and masking—lets a standard BERT architecture and the plain MLM objective reach or surpass state-of-the-art results, clarifying the real levers for progress in language model pretraining.
