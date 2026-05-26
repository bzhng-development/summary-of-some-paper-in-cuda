# Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification

**URL:** [https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)

## 🎯 Pitch

This paper audits three commercial gender classification systems—Microsoft, IBM, and Face++—for intersectional accuracy disparities across phenotypic subgroups.

---

## 1. Executive Summary

This paper audits three commercial gender classification systems—Microsoft, IBM, and Face++—for intersectional accuracy disparities across phenotypic subgroups. Using the Fitzpatrick Skin Type classification system, the authors characterize the skin-type and gender composition of two existing face benchmarks (IJB-A and Adience), find them overwhelmingly lighter-skinned (79.6% and 86.2% respectively), and introduce the **Pilot Parliaments Benchmark (PPB)**, a 1,270-subject dataset balanced by gender and skin type. An intersectional evaluation of the three classifiers on PPB reveals that darker-skinned females are the most misclassified group, with error rates up to 34.7%, while lighter-skinned males experience maximum error rates of only 0.8%—a 34.4 percentage-point gap between best- and worst-classified subgroups—establishing that aggregate accuracy metrics conceal severe subgroup performance disparities which only emerge when evaluation is disaggregated by gender, skin type, and their intersection simultaneously.

## 2. Context and Motivation

### The Specific Problem: Aggregate Metrics Conceal Subgroup Failures

At the time of this paper's publication, commercial facial analysis systems were being deployed across high-stakes domains—law enforcement, hiring, healthcare—with remarkably little public scrutiny of their performance across different populations. The default evaluation paradigm was to report **aggregate accuracy**: a single number like "97.35% accuracy on the LFW benchmark" (Taigman et al., 2014) that suggested near-human performance. This paper identifies a fundamental blind spot in that practice: aggregate metrics, by averaging across all test subjects, can hide catastrophic failure on minority subgroups. A system that classifies lighter-skinned males with 99.2% accuracy and darker-skinned females with 65.3% accuracy can still report "90% overall accuracy," a figure that would suggest the system is suitable for all populations when it is not.

This is not a hypothetical concern. The paper demonstrates the mechanism concretely: darker females constitute only 21.3% of the PPB benchmark but account for between 61.0% and 72.4% of all classification errors across the three evaluated systems. A developer looking only at an aggregate 93.7% accuracy (Microsoft's overall score on PPB) would have no indication that one specific demographic subgroup—darker-skinned females—had a 20.8% error rate, more than 20 times higher than lighter-skinned males at 0.0%. The gap between the problem the paper diagnoses (hidden subgroup failures) and the problem most developers thought they were solving (general accuracy improvement) is the central tension that motivates every aspect of the study.

### Why the Problem Matters: Real-World Stakes

The paper builds its case for urgency along multiple dimensions, each escalating the consequences of inaccurate facial analysis:

**Law enforcement and surveillance.** Section 1 opens with the concrete risk of wrongful accusation: "someone could be wrongfully accused of a crime based on erroneous but confident misidentification of the perpetrator from security video footage analysis." This is not speculative. The paper cites Garvie et al. (2016)'s finding that at least 117 million Americans are included in law enforcement face recognition networks, and that African-American individuals are more likely to be stopped and subjected to face recognition searches. If those face recognition systems inherit gender classification biases—and gender classification is often a preprocessing step in face recognition pipelines—then the subgroup error patterns identified in this paper compound existing racial disparities in policing.

**Healthcare applications.** The paper draws a direct parallel to biased clinical trials (Section 1): "similar to the well-documented detrimental effects of biased clinical trials, biased samples in AI for health care can result in treatments that do not work well for many segments of the population." Dermatology AI systems (Esteva et al., 2017) that classify skin cancer from images but are trained on datasets lacking diversity in skin type, hair, and skin thickness will produce unmeasured—and potentially dangerous—performance disparities for darker-skinned patients. This parallel is powerful because it connects algorithmic fairness to an established medical ethics framework where the consequences of biased sampling (misdiagnosis, ineffective treatment) are well understood, making the gap in computer vision accountability more stark by comparison.

**Pipeline effects and cascading errors.** A subtler but crucial point the paper raises is that even systems not directly performing high-stakes tasks can propagate harm through downstream pipelines. Section 1 states: "while face recognition software by itself should not be trained to determine the fate of an individual in the criminal justice system, it is very likely that such software is used to identify suspects. Thus, an error in the output of a face recognition algorithm used as input for other tasks can have serious consequences." This means the accountability burden extends beyond obviously consequential systems: any component in a decision pipeline inherits responsibility for its errors, and the benchmarks used to validate those components must reflect that responsibility.

### Where Existing Benchmarks Fall Short

The paper identifies two canonical benchmarks that were widely used at the time and shows systematically why neither is suitable for intersectional performance evaluation:

**IJB-A (2015): Government-released but insufficiently balanced.** Released by IARPA and NIST as an attempt at geographic diversity, the IJB-A dataset was advertised as a benchmark that "limits bias" by not using face detectors to select images (Klare et al., 2015). However, the paper's skin-type annotation reveals that despite geographic diversity, IJB-A is 79.6% lighter-skinned and contains only 4.4% darker-skinned females (Figure 3, Table 3). This means the dataset's diversity claim—based on geography—does not translate to phenotypic diversity. Countries can have phenotypically diverse populations, and geographic sampling alone does not guarantee representation across skin types. The paper exposes this gap directly: a benchmark can claim to be geographically diverse while still being overwhelmingly light-skinned, making it useless for detecting performance disparities on darker-skinned subjects.

**Adience (2014): The gender classification standard with severe skew.** Released specifically as a gender and age classification benchmark (Levi and Hassner, 2015b), Adience contains 2,284 unique subjects—a seemingly large and useful sample. But the paper's annotation reveals that 86.2% of those subjects are lighter-skinned, and only 13.8% are darker-skinned (Table 3). Even more alarming from an intersectional perspective: only 4.4% of subjects are darker-skinned and female (Figure 3), while lighter males constitute 41.6%. This means that any gender classifier trained or evaluated on Adience effectively treats darker-skinned females as a negligible population—the worst possible scenario for training a system to serve that group well.

**The face-detection feedback loop.** The paper notes an additional structural problem with benchmark construction in Section 2: "most large-scale attempts to collect visual face datasets rely on face detection algorithms to first detect faces... Any systematic error found in face detectors will inevitably affect the composition of the benchmark." If face detectors are themselves biased (and this paper's results on gender classification strongly suggest they might be), then datasets constructed using those detectors will systematically underrepresent the groups the detectors miss. This creates a self-reinforcing cycle: biased detectors produce biased datasets, which train biased classifiers, which are used to filter the next generation of datasets. The paper doesn't resolve this chicken-and-egg problem but naming it explicitly motivates the need for benchmarks—like PPB—that are constructed without automated face-detection filtering.

### How Prior Evaluation Methods Obscured Disparities

Beyond specific datasets, the paper identifies deeper methodological failures in how facial analysis systems were evaluated:

**Race and ethnicity labels are unstable proxies.** Section 3.1 makes a careful argument for why traditional demographic categories are insufficient for visual benchmark auditing. Race labels, while suitable for some forms of algorithmic auditing (e.g., recidivism prediction), face two specific limitations for image-based tasks: (1) "subjects' phenotypic features can vary widely within a racial or ethnic category"—lighter-skinned Black individuals would not reveal how a system performs on darker-skinned Black individuals—and (2) "racial and ethnic categories are not consistent across geographies: even within countries these categories change over time." The paper cites the binary "Caucasian/non-Caucasian" categorization used by Farinella and Dugelay (2012), which claimed ethnicity had no effect on gender classification—a conclusion the paper's more granular analysis fatally undermines. By collapsing all non-Caucasian phenotypes into a single category, prior work simply lacked the resolution to detect the performance cliff the paper documents.

**NIST evaluations used country-of-origin as a skin-type proxy.** The National Institute of Standards and Technology's gender classification evaluation (Ngan et al., 2015) reported worse performance on female-labeled faces than male-labeled faces, which the paper acknowledges as an important precedent. But the paper also identifies a critical gap in NIST's methodology: "none of the 10 locations used in the study were in Africa or the Caribbean where there are significant Black populations." Using country of origin as an ethnicity proxy is a weak substitute for phenotypic labeling. A study that evaluates on subjects from, say, Japan or Brazil may miss entirely the performance profile on darker-skinned individuals from African nations. The paper's Fitzpatrick labeling approach is explicitly designed to close this gap by providing a phenotype-based measure that doesn't rely on geographic or ethnic proxies.

**No prior work evaluated the intersection of gender and skin type simultaneously.** This is the paper's central methodological contribution and its sharpest critique of prior practice. Existing studies evaluated gender accuracy or skin-type accuracy independently. Section 4.5 makes the case explicitly: "Though helpful in seeing systematic error, gender and skin type analysis by themselves do not present the whole story. Is misclassification distributed evenly amongst all females? Are there other factors at play?" The paper's answer, backed by data, is no: darker males (6.0% error for Microsoft) and lighter females (1.7% error for Microsoft) both outperform darker females dramatically (20.8% error for Microsoft), meaning the performance penalty is multiplicative across the two attributes rather than additive. Neither a gender-only analysis nor a skin-type-only analysis would capture this interaction.

### How This Paper Positions Itself

The paper positions itself at the intersection of two existing research traditions—algorithmic fairness and computer vision benchmarking—and argues that neither has adequately addressed the other:

**A fairness intervention in computer vision.** While the algorithmic fairness community had developed sophisticated frameworks for measuring and mitigating discrimination (Hardt et al., 2016b,a; Kilbertus et al., 2017) and had conducted audits of non-visual algorithmic systems (Angwin et al., 2016 on recidivism prediction), the paper notes that "only a handful of works have done this analysis for computer vision" (Section 1). The paper explicitly aims to extend the algorithmic auditing methodology—demographic subgroup analysis, transparency about benchmark composition, accountability for performance disparities—to a domain where it was notably absent: commercial visual classification systems.

**A benchmarking intervention in fairness.** Simultaneously, the paper argues that fairness work needs better benchmarks. The creation of PPB is not just about this one study; it is offered as infrastructure for future research. Section 5 defines downstream ambitions: "future work should explore intersectional phenotypic and demographic error analysis of facial detection, identification and verification." The paper is establishing a template—balanced benchmark construction + intersectional subgroup reporting—that it argues should become standard practice across computer vision, not a one-time finding.

**From race to phenotype.** The paper's most distinctive positioning move is its argument that **phenotypic labels** (specifically skin type) are more scientifically rigorous and practically useful for visual fairness auditing than racial or ethnic labels. Section 3.1 justifies this with both scientific reasoning (the Fitzpatrick scale is dermatologist-validated and tied to skin cancer risk assessment) and practical reasoning (skin type is directly relevant to how camera sensors capture images, since "default camera settings are calibrated to expose lighter-skinned individuals" (Roth, 2009)). This positions the paper's methodology not as a rejection of demographic auditing but as a refinement—using labels that are more stable, more visually precise, and more causally connected to potential failure modes than race or ethnicity alone.

**Transparency and accountability defined.** The paper concludes with two operational definitions meant to guide future practice (Section 5): "transparency as providing information on the demographic and phenotypic composition of training and benchmark datasets" and "accountability as reporting algorithmic performance on demographic and phenotypic subgroups and actively working to close performance gaps where they arise." These definitions are deliberately narrower than full sociotechnical frameworks (the paper notes they "do not focus on" consent and redress mechanisms), but they translate abstract fairness principles into concrete, auditable actions: disclose your data composition, report subgroup accuracy, fix the gaps. The entire study is an existence proof that this reporting is both possible and revealing.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is primarily an **empirical audit study** that constructs a measurement apparatus to detect and quantify intersectional accuracy disparities in commercial facial analysis systems. The "system" being built is not a new classifier, but rather a **benchmarking and evaluation framework** composed of a carefully curated dataset (PPB), a skin-type labeling protocol (Fitzpatrick scale), and an intersectional error analysis methodology. The problem it solves is the invisibility of subgroup failures: existing evaluation paradigms reported only aggregate accuracy, which could show 90%+ overall performance while concealing 34.7% error rates on darker-skinned females. The shape of the solution is a measurement instrument—balanced benchmark + phenotype-based annotation + disaggregated subgroup reporting—that makes these hidden disparities visible and quantifiable.

### 3.2 Big-Picture Architecture

The evaluation framework has four major components organized in a pipeline:

1. **Dataset Selection and Annotation Protocol** — Identifies existing benchmarks (IJB-A, Adience) and a newly constructed dataset (PPB), applies the Fitzpatrick six-point skin type classification system and binary gender labels to every subject, and assigns each subject to one of four intersectional subgroups (darker female, darker male, lighter female, lighter male) for downstream analysis.

2. **Pilot Parliaments Benchmark (PPB) Construction** — Curates 1,270 images of parliamentarians from six countries (three African: Rwanda, Senegal, South Africa; three European: Iceland, Finland, Sweden) selected specifically to achieve gender parity and skin-type balance, producing a dataset where no subgroup is numerically negligible.

3. **Commercial API Querying Infrastructure** — Submits PPB images to three commercial gender classification APIs (Microsoft Cognitive Services Face API, IBM Watson Visual Recognition, Face++) at a fixed point in time (April–May 2017), collecting binary gender predictions and, where available, confidence scores.

4. **Intersectional Error Analysis** — Computes true positive rate, false positive rate, positive predictive value, and error rate independently for each of the four intersectional subgroups, then compares these metrics across subgroups to reveal performance gaps that would be invisible in aggregate.

Information flows sequentially: subject images are collected → each image receives skin type and gender labels from human annotators (with a dermatologist providing definitive labels) → labelled images are submitted to each commercial API → API predictions are compared to ground-truth labels → per-subgroup accuracy metrics are computed → disparities are quantified as error rate differences between best- and worst-performing subgroups.

### 3.3 Roadmap for the Deep Dive

- **First**, the Fitzpatrick skin type labeling protocol — how skin types are defined, why this scale over alternatives, and how the paper handles its limitations — because skin type is the primary phenotypic axis and every subsequent decision depends on its validity.
- **Second**, the intersectional subgrouping scheme — how individual subjects are assigned to one of four analytic categories (darker female, darker male, lighter female, lighter male) — since the paper's central claim is that intersectional analysis reveals disparities invisible to univariate analyses.
- **Third**, the PPB dataset construction — country selection rationale, image collection, annotation methodology, and resulting subgroup distributions — because PPB is the measurement instrument whose composition determines what disparities can be detected.
- **Fourth**, the commercial API evaluation protocol — how images are submitted, what outputs are collected, what thresholds and decision rules the APIs impose, and what metrics are computed — since these choices determine whether the measured disparities reflect classifier behavior or evaluation artifacts.
- **Fifth**, the intersectional error decomposition — how aggregate error rates are mathematically partitioned across subgroups and what patterns this reveals — because this is the analytic procedure that produces the paper's headline findings.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This paper is an **empirical audit study** whose core technical contribution is the construction of a measurement framework that can detect intersectional accuracy disparities in commercial gender classification systems. The framework has no learnable parameters — it is purely an evaluation instrument — but its design involves careful methodological choices at every stage: phenotype labeling, subgroup definition, dataset construction, API querying, and metric computation. I will walk through each component in the order that information flows through the system.

---

#### Fitzpatrick Skin Type Labeling Protocol

**What the Fitzpatrick scale is.** The Fitzpatrick skin type classification system, originally developed by dermatologist Thomas B. Fitzpatrick in 1975 and formalized in 1988 (TB, 1988), is a six-point ordinal scale that classifies human skin based on its response to ultraviolet (UV) light exposure. The six types are:

- **Type I**: Pale white skin; always burns, never tans.
- **Type II**: White skin; burns easily, tans minimally.
- **Type III**: White to light brown skin; burns moderately, tans gradually.
- **Type IV**: Olive to moderate brown skin; burns minimally, tans well.
- **Type V**: Brown to dark brown skin; very rarely burns, tans profusely.
- **Type VI**: Dark brown to black skin; never burns, tans deeply.

The scale is the dermatologist's gold standard for assessing skin cancer risk and determining appropriate UV treatment dosages. The paper adopts it for a different purpose — benchmark diversity characterization — but inherits the scale's scientific validation. Specifically, the scale's clinical provenance means there is an established body of medical literature supporting its reliability when applied by trained dermatologists, which addresses one potential objection: that skin type labeling from images is inherently subjective and unreliable.

**Why Fitzpatrick over race or ethnicity labels.** The paper provides a detailed justification in Section 3.1 that addresses two distinct failure modes of race-based labeling for visual benchmarks:

First, **within-category phenotypic variation**: "subjects' phenotypic features can vary widely within a racial or ethnic category. For example, the skin types of individuals identifying as Black in the US can represent many hues. Thus, facial analysis benchmarks consisting of lighter-skinned Black individuals would not adequately represent darker-skinned ones." This is not just a theoretical concern — it is a measurement problem. If a benchmark claims to evaluate performance on Black subjects but primarily contains lighter-skinned Black individuals, it will systematically underestimate disparities affecting darker-skinned individuals within that same racial category. Using skin type directly closes this gap by measuring the phenotype of interest — skin color — rather than relying on a social category (race) that is correlated with skin color but far from identical to it.

Second, **geographic instability of racial categories**: "racial and ethnic categories are not consistent across geographies: even within countries these categories change over time." The paper notes this is a particular problem for international benchmarks like IJB-A, which aims to be geographically diverse. An individual classified as "White" in Brazil may be classified differently in the United States; an individual classified as "Coloured" in South Africa has no equivalent category in many other countries' racial taxonomies. Skin type, by contrast, is a physical measurement that is (in principle) independent of the social context in which the measurement is made.

Additionally, the paper identifies a **sensor-specific causal pathway** that makes skin type directly relevant to facial analysis failures independent of fairness considerations: "default camera settings are calibrated to expose lighter-skinned individuals (Roth, 2009). Poorly exposed images that result from sensor optimizations for lighter-skinned subjects or poor illumination can prove challenging for automated facial analysis." This is a crucial point — the paper is not just arguing that skin type is a better demographic label for fairness auditing; it is arguing that skin type is mechanistically linked to the technical quality of the input images through camera sensor design, meaning classification failures on darker-skinned subjects may be partly attributable to lower-quality input data rather than (or in addition to) biased training data or model architectures.

**The labeling procedure for existing benchmarks.** For IJB-A (500 subjects) and Adience (2,284 subjects), one author labeled each subject's skin type using a single reference image per subject. The paper acknowledges the coarseness of this approach implicitly — it is a single annotator, a single image per subject, with no inter-annotator reliability reported — but frames it as a screening step rather than a definitive characterization. The purpose was to "determine if a new benchmark was needed" (Section 3.5), and the finding that these benchmarks are 79.6% and 86.2% lighter-skinned respectively (Table 3) was sufficiently clear-cut that the annotation served its purpose even if individual labels had some error.

**The labeling procedure for PPB.** For the newly constructed Pilot Parliaments Benchmark, the annotation protocol was substantially more rigorous:

1. Three annotators, including both authors, independently provided Fitzpatrick skin type labels for each of the 1,270 subjects.
2. A board-certified surgical dermatologist (Dr. Helen Raynham, acknowledged in the paper) reviewed and provided the **definitive labels** — meaning the dermatologist's judgment served as ground truth, and the other annotators' labels were presumably used to flag ambiguous cases for dermatologist review.
3. Labels were assigned based on visible facial skin in the parliamentary portraits, not on self-reported skin type or Fitzpatrick questionnaire responses (the clinical gold standard involves both a spectrophotometer reading and a patient questionnaire about sun sensitivity).

The dermatologist's involvement is a distinctive strength of the paper's methodology. By having a medical expert who routinely classifies skin types in clinical practice provide the definitive labels, the paper substantially reduces the risk that its findings are artifacts of unreliable annotation. This is expensive and difficult to replicate — which is precisely why the paper emphasizes it: it raises the bar for what constitutes rigorous phenotypic benchmarking.

**The lighter/darker aggregation.** For the primary analysis, the paper does not report results for all six Fitzpatrick types individually. Instead, it aggregates Types I–III into a "lighter" category and Types IV–VI into a "darker" category. The paper provides an explicit justification for this aggregation (Section 3.5): "The skin types are aggregated to account for potential off-by-one errors since the skin type is estimated using images instead of employing a standard spectrophotometer and Fitzpatrick questionnaire."

This is an honest methodological concession. The clinical Fitzpatrick scale was designed to be administered with both instrumental measurement (spectrophotometer) and patient self-report (sun sensitivity questionnaire). Applying it retrospectively to photographs introduces ambiguity — a Type III subject might be labeled Type IV by a different annotator or under different lighting, and vice versa. By drawing the boundary at the III/IV transition (the traditional divide between "White" and "olive-to-brown" skin in dermatological practice), the paper ensures that off-by-one errors are unlikely to cause a subject to cross the lighter/darker boundary. The trade-off is reduced granularity: the paper cannot distinguish, say, whether Type VI subjects are more misclassified than Type IV subjects. But this is a reasonable trade-off given the annotation constraints, and the paper reports per-Fitzpatrick-type results in supplementary Table 2 for readers who want the full disaggregation.

**A limitation the paper does not fully address.** The Fitzpatrick scale, as the paper acknowledges (Section 3.5), "is skewed towards lighter skin and has three categories that can be applied to people perceived as White." Types I, II, and III all describe variations of light skin (pale white, white, white-to-light-brown), while Types IV, V, and VI must cover the entire global range from olive to the darkest possible skin tones. This means the scale has finer discrimination in the lighter range than the darker range, which is exactly the kind of measurement bias the paper critiques in other contexts. The paper acknowledges this asymmetry but does not propose an alternative scale — it treats the Fitzpatrick system as the best available scientifically-validated starting point, and its call for future work implicitly includes the development of more balanced phenotypic classification systems.

---

#### Intersectional Subgroup Definition

**The four analytic categories.** Every subject in PPB is assigned to exactly one of four intersectional subgroups defined by crossing the binary gender variable (female/male) with the binary skin-type variable (lighter/darker):

- **Darker Female (DF)**: subjects labeled female with Fitzpatrick skin type IV, V, or VI.
- **Darker Male (DM)**: subjects labeled male with Fitzpatrick skin type IV, V, or VI.
- **Lighter Female (LF)**: subjects labeled female with Fitzpatrick skin type I, II, or III.
- **Lighter Male (LM)**: subjects labeled male with Fitzpatrick skin type I, II, or III.

These four categories form a complete partition of the dataset: every subject belongs to exactly one subgroup, and the subgroups are mutually exclusive. This is the standard structure of an intersectional analysis — rather than evaluating gender disparities independently (male vs. female) and skin-type disparities independently (lighter vs. darker), the analysis treats the combination of attributes as the unit of analysis.

**Why intersectional analysis is necessary.** The paper's methodological argument (Section 4.5) is that univariate analyses — gender alone or skin type alone — produce misleading averages. A gender-only analysis would report female error rates that average together darker females (who are heavily misclassified) and lighter females (who are classified relatively well). A skin-type-only analysis would report darker-subject error rates that average together darker females (heavily misclassified) and darker males (classified relatively well). In both cases, the worst-performing subgroup — darker females — is hidden by averaging with a better-performing subgroup that shares one attribute but not both.

This is the "interaction effect" argument applied to fairness measurement: if misclassification depends on both gender and skin type *multiplicatively* (or at least non-additively), then no single-axis analysis can capture the full pattern. The paper's empirical results confirm this is the case — darker females have dramatically higher error rates than either darker males or lighter females alone would predict, meaning the combination of attributes produces a distinct disadvantage beyond what either attribute produces independently.

**What the subgroups do NOT capture.** The paper is explicit that "this reductionist view of gender does not adequately capture the complexities of gender or address transgender identities" (Section 3.4). The binary gender labels are imposed by the commercial APIs (all three provide only male/female classifications) and by the parliamentary data (where subjects are listed with gendered titles and prefixes). The paper uses these binary labels because they are the labels the systems under audit use, not because they endorse a binary view of gender. This is an important methodological distinction: the paper audits existing systems on their own terms — using the categories those systems claim to classify — while simultaneously flagging that those categories are themselves inadequate.

**Subgroup sizes and their implications for statistical reliability.** The distribution of subjects across the four subgroups (Table 2, total row) is:

- Darker Female: 21.3% (270 subjects)
- Darker Male: 25.0% (318 subjects)
- Lighter Female: 23.3% (296 subjects)
- Lighter Male: 30.3% (385 subjects)

The "All Subjects" row in Table 2 gives the full dataset size: n = 1,270. The percentages translate to the approximate counts shown above (I compute these from the percentages: 1,270 × 0.213 ≈ 270, etc.).

This distribution is far from perfectly balanced — lighter males are substantially overrepresented relative to an ideal equal split (25% per group) — but it achieves the paper's stated goal: no subgroup is so small that its error rate estimates are unreliable. The smallest subgroup (darker females, ~270 subjects) is large enough that a 20.8% error rate (Microsoft) corresponds to roughly 56 misclassified subjects, which is a sufficiently large count that the estimate is unlikely to be purely noise. This contrasts sharply with IJB-A, where darker females constitute only 4.4% of 500 subjects (22 subjects total), making any subgroup-specific error rate estimate based on IJB-A alone extremely noisy.

---

#### PPB Dataset Construction

**The country selection rationale.** The paper selects six countries — three African (Rwanda, Senegal, South Africa) and three European (Iceland, Finland, Sweden) — based on two explicit criteria and one implicit criterion:

**Criterion 1: Gender parity in national parliaments.** The paper uses Inter-Parliamentary Union rankings to identify countries with the highest proportion of women in parliament. Rwanda ranks first globally. Nordic countries (Iceland, Finland, Sweden) are also highly ranked. By selecting from this list, the paper ensures that parliamentary photo galleries contain roughly equal numbers of men and women, eliminating the gender skew that would plague datasets sourced from institutions with severe gender imbalance.

**Criterion 2: Skin type diversity.** The African countries are selected to provide darker-skinned subjects, and the Nordic countries to provide lighter-skinned subjects. The paper references Figure 2 (a skin color distribution map from Encyclopedia Britannica) to justify the geographic clustering of skin types: "African countries typically have darker-skinned individuals whereas Nordic countries tend to have lighter-skinned citizens." However, the paper immediately adds an important caveat: "Colonization and migration patterns nonetheless influence the phenotypic distribution of skin type and not all Africans are darker-skinned. Similarly, not all citizens of Nordic countries can be classified as lighter-skinned." This caveat is empirically borne out in the results: South Africa's parliament includes both lighter-skinned (20.8%) and darker-skinned (79.2%) subjects (Table 2, South Africa row), reflecting the country's diverse population.

**Criterion 3 (implicit): Availability of standardized portraits under non-restrictive licenses.** Parliamentary websites typically provide official portraits of members under government or Creative Commons licenses, taken under standardized conditions (consistent pose, lighting, and background). This is what makes PPB "highly constrained" (Section 3.3): the images have relatively little variation in pose, illumination, and expression compared to in-the-wild benchmarks like Adience. The paper explicitly notes that it "intentionally chose an optimistic sample of constrained images" (Section 4.7) to avoid the confound where pose and illumination variation could explain performance disparities. By using images where technical quality is relatively uniform, the paper isolates skin type and gender as the primary axes of variation.

**Country-by-country composition.** Table 2 provides the decomposition of PPB by country, which reveals substantial heterogeneity within both the African and European blocs:

- **Rwanda** (n=75): 60.0% female, 40.0% male, **100% darker-skinned**. The small sample size (75 subjects, the smallest of any country) reflects Rwanda's smaller parliament. All subjects are darker-skinned, meaning Rwanda contributes exclusively to the DF and DM subgroups.

- **Senegal** (n=149): 43.0% female, 57.0% male, **100% darker-skinned**. Like Rwanda, all Senegalese parliamentarians are classified as darker-skinned, contributing only to the DF and DM subgroups.

- **South Africa** (n=437, the largest country subset): 41.4% female, 58.6% male, **79.2% darker, 20.8% lighter**. South Africa is the most phenotypically diverse country in PPB and serves a crucial methodological role. Because it contains substantial numbers of both lighter and darker subjects taken under uniform imaging conditions (the paper notes South African images "had the most consistent pose and illumination"), it functions as a within-country control: if disparities exist in the South African subset alone, they cannot be attributed to between-country differences in image quality or photographic conventions.

- **Sweden** (n=349): 46.7% female, 53.3% male, **4.9% darker, 95.1% lighter**.

- **Finland** (n=197): 42.6% female, 57.4% male, **1.0% darker, 99.0% lighter**.

- **Iceland** (n=63): 47.6% female, 52.4% male, **0.0% darker, 100% lighter**.

The European countries contribute almost exclusively to the LF and LM subgroups, with Sweden providing a small number of darker-skinned subjects (4.9% of 349 ≈ 17 subjects). The combined Africa/Europe split (Table 2, Africa and Europe rows) shows the intended design: 86.2% of African subjects are darker-skinned, and 96.9% of European subjects are lighter-skinned.

**Image characteristics.** Table 1 compares PPB's image properties to IJB-A and Adience:

- **Number of subjects**: PPB has 1,270, compared to 500 for IJB-A and 2,284 for Adience. PPB is larger than IJB-A but smaller than Adience, placing it in a middle ground for benchmark size.

- **Average interpupillary distance (IPD)**: 63 pixels for PPB. IPD is a standard measure of face resolution in biometrics — it is the distance between the centers of the two eyes in pixels. 63 pixels is a reasonable resolution for facial analysis; many face recognition systems require a minimum IPD of 40–60 pixels. The paper does not report IPD for IJB-A or Adience (marked as "-" in the table), making direct comparison impossible, but notes IJB-A has a minimum bounding box size of 36 pixels.

- **Average bounding box size**: 141 pixels for PPB. The bounding box is the square region containing the face. The paper does not provide Adience or IJB-A bounding box sizes, but notes IJB-A has a minimum of 36 pixels.

- **Image dimensions**: PPB images range from 160–590 pixels in width and 213–886 pixels in height. Adience images are uniformly 816 × 816 pixels (a standardized crop size), while IJB-A dimensions are not reported.

**Annotation process.** For PPB, gender labels were determined from three sources (Section 3.4): "the name of the parliamentarian, gendered title, prefixes such as Mr or Ms, and the appearance of the photo." This means the gender annotation relies primarily on the parliamentarians' self-presentation in official government records (their names and titles) rather than purely on visual appearance. This is a stronger methodology than appearance-only gender labeling because it uses administrative ground truth rather than annotator perception.

**A potential confound the paper addresses.** Section 4.7 directly engages with the possibility that between-country differences in image quality could explain the observed disparities: "In PPB, the European parliamentary images tend to be of higher resolution with less pose variation when compared to images from African parliaments." If darker-skinned subjects appeared predominantly in lower-quality images, then poor image quality rather than skin type could drive the higher error rates.

The paper argues against this confound using the South African subset: "The South African parliament, however, has comparable image resolution and has the largest skin type spread of all the parliaments. Lighter subjects makeup 20.8% (n=91) of the images, and darker subjects make up the remaining 79.2% (n=346) of images." The South Africa-only analysis (Table 5) replicates the overall pattern — darker females are most misclassified, lighter males least — within a single country where imaging conditions are uniform. The paper concludes: "Thus, we conclude that variation in performance due to the image characteristics of each country does not fully account for the differences in misclassification rates between intersectional subgroups."

The paper then offers an alternative explanation that is more measured than the strong causal claim: "darker skin alone may not be fully responsible for misclassification. Instead, darker skin may be highly correlated with facial geometries or gender display norms that were less represented in the training data of the evaluated classifies." This is an important methodological honesty: the paper can measure *that* darker-skinned females are more misclassified, but it cannot determine *whether* the causal mechanism is skin reflectance properties (affecting sensor exposure), facial feature distributions (affecting classifier decision boundaries), or gender expression norms (affecting what the classifier learns to associate with "female" vs. "male").

---

#### Commercial API Querying Protocol

**Classifier selection rationale.** The paper selects three commercial gender classification systems (Section 4.2):

1. **Microsoft Cognitive Services Face API**: Chosen because Microsoft had made large AI investments and captured significant market share in the machine learning services domain. At the time of evaluation, the service description stated it used "advanced statistical algorithms" that "may not always be 100% precise" (Microsoft API Reference).

2. **IBM Watson Visual Recognition**: Chosen for similar market-share reasons. IBM described its service as using "deep learning-based algorithms" (IBM API Reference). Critically, IBM was the only provider that returned confidence scores alongside binary gender classifications, providing additional information for analysis.

3. **Face++**: Chosen because it was headquartered in China, and "previous studies have shown that face recognition systems developed in Western nations and those developed in Asian nations tend to perform better on their respective populations" (Phillips et al., 2011). The hypothesis being tested — though not explicitly stated as a hypothesis — was whether a Chinese-developed system would show different demographic error patterns than the US-based Microsoft and IBM systems.

Google was excluded because, at the time of evaluation (April–May 2017), Google did not provide a publicly available gender classification API — though its Face API for Android existed, it did not expose gender as a standalone feature accessible to researchers.

**What the APIs output.** The paper is careful to document exactly what each API returns, because this constrains what can be measured:

- **Microsoft and Face++**: Output only a **single binary label** — "male" or "female" — with no associated probability or confidence score. This means the API applies an internal threshold to transform a continuous prediction (which presumably exists inside the model) into a discrete classification, and neither the threshold nor the pre-threshold probability is exposed to users.

- **IBM**: Outputs both a **binary label and a confidence score** (between 0 and 1). Figure 4 plots the distribution of these confidence scores across the four intersectional subgroups. The scores are near 1 for lighter males and lighter females, while they range from approximately 0.75 to 1 for darker females, indicating IBM's classifier is both more *error-prone* and less *confident* on darker-skinned female subjects.

**The threshold opacity problem.** The paper raises a significant methodological concern about the non-probabilistic APIs (Section 4.6): "All 3 evaluated APIs only provide gender classifications, they do not output probabilities associated with the likelihood of being a particular gender. This indicates that companies are choosing a threshold which determines the classification: if the prediction probability is greater than this threshold, the image is determined to be that of a male (or female) subject, and viceversa if the probability is less than this number."

This matters because the choice of classification threshold determines the trade-off between true positive rates (TPR) and false positive rates (FPR) for each subgroup. A higher threshold for "male" classification means fewer subjects are classified as male overall, reducing false positives (females misclassified as male) but increasing false negatives (males misclassified as female). If the threshold is chosen to optimize aggregate accuracy, it may implicitly prioritize the majority subgroup's error profile. Without access to the raw probabilities, external auditors cannot assess whether subgroup disparities reflect fundamental differences in classifier separability or simply threshold choices that happen to disadvantage certain groups.

The paper frames this as an accountability deficit: "By having APIs that fail to provide the ability to adjust these thresholds, they are limiting users' ability to pick their own TPR/FPR trade-off." This is a specific, actionable critique: commercial APIs should expose prediction probabilities (not just hard labels) so that downstream users can set thresholds based on their own fairness-accuracy trade-offs, rather than having a single opaque threshold imposed by the vendor.

**Temporal specificity of the audit.** The paper specifies the evaluation timeframe: April and May 2017. This matters because commercial APIs are continuously updated behind the scenes — models are retrained, thresholds adjusted, training data expanded — without public changelogs. The results represent a snapshot of three specific API versions at one moment in time. The paper cannot claim that these disparities persist indefinitely, and indeed the publicity the paper received when published in 2018 may have prompted the vendors to address these issues. The temporal specificity is both a limitation (the results may not generalize to later API versions) and a methodological strength (it enables reproducibility and accountability at a fixed point in time).

**API documentation opacity.** The paper documents what each company *did not* disclose (Section 4.2): "The description of classification methodology lacked detail and there was no mention of what training data was used." None of the three commercial classifiers reported performance metrics on existing gender estimation benchmarks in their provided documentation. Face++'s terms of use "explicitly disclaim any warranties of accuracy." IBM provided confidence scores but "did not report how any metrics like true positive rates (TPR) or false positive rates (FPR) were balanced."

This documentation of *absence* — what the companies failed to disclose — is itself part of the paper's methodological contribution. By systematically cataloguing what information commercial vendors do and do not provide, the paper makes the case that the current state of algorithmic transparency is grossly insufficient for accountability. The paper's own methodology — reporting subgroup TPR, FPR, PPV, and error rates — serves as a template for what responsible disclosure would look like.

---

#### Evaluation Metrics and Intersectional Error Decomposition

**Metric definitions.** The paper uses four standard binary classification metrics, computed independently for each of the four intersectional subgroups (Table 4):

For each metric, I will define it in terms of the gender classification task where the positive class is "female" and the negative class is "male" (the paper's analysis is symmetric — classification of males and females are both evaluated — but using one class simplifies notation).

- **True Positive Rate (TPR, also called recall or sensitivity)**: For the female class, TPR is the proportion of actual female subjects correctly classified as female. For the male class, TPR is the proportion of actual male subjects correctly classified as male. More formally, for a given subgroup $s$:

$$\text{TPR}_s = \frac{\text{Number of subjects in subgroup } s \text{ correctly classified}}{\text{Total number of subjects in subgroup } s}$$

where "correctly classified" means the predicted gender matches the ground-truth gender label. This is the standard definition of per-class accuracy.

- **Error Rate**: The complement of TPR:

$$\text{Error Rate}_s = 1 - \text{TPR}_s = \frac{\text{Number of subjects in subgroup } s \text{ misclassified}}{\text{Total number of subjects in subgroup } s}$$

This is the metric the paper primarily uses when reporting disparities, because it directly communicates the probability that a member of a given subgroup will experience a classification failure.

- **Positive Predictive Value (PPV, also called precision)**: For the female class, PPV is the proportion of subjects *predicted* to be female who are *actually* female. That is, when the classifier says "female," how often is it right?

$$\text{PPV}_s^{\text{female}} = \frac{\text{Number of subjects in subgroup } s \text{ correctly classified as female}}{\text{Total number of subjects in subgroup } s \text{ predicted as female}}$$

- **False Positive Rate (FPR)**: The complement of precision for the negative class. For females, FPR is the proportion of actual male subjects incorrectly classified as female. For males, FPR is the proportion of actual female subjects incorrectly classified as male.

$$\text{FPR}_s^{\text{female}} = \frac{\text{Number of male subjects in subgroup } s \text{ misclassified as female}}{\text{Total number of male subjects in subgroup } s}$$

**Why report all four metrics rather than just accuracy.** The paper's approach follows the NIST gender classification evaluation precedent (Ngan et al., 2015) but extends it. NIST reported overall accuracy, male accuracy, and female accuracy — essentially TPR for each class. The paper adds PPV and FPR because these metrics capture different aspects of classifier behavior that matter for different use cases:

- A high TPR (most females correctly classified) combined with a high FPR (many males also classified as female) indicates a classifier that is biased toward predicting "female" regardless of the input — it catches most females but also generates many false alarms.

- A high PPV for females combined with a low TPR for females indicates a classifier that is conservative about predicting "female" — when it does predict female, it is usually right, but it misses many actual females.

- A system used for surveillance (where the cost of a false positive might be wrongful suspicion) has different requirements than a system used for demographic analytics (where the cost is statistical inaccuracy). Reporting all four metrics allows downstream users to evaluate fitness for their specific purpose.

**The intersectional error decomposition.** The paper's key analytic move is computing these metrics *separately for each intersectional subgroup* rather than computing them for gender alone or skin type alone. The mathematical operation is straightforward — it is simply stratifying the metric computation — but the conceptual implication is profound: each subgroup is treated as its own evaluation unit, with its own TPR, FPR, PPV, and error rate. There is no pooling across subgroups.

The paper then quantifies intersectional disparities as **pairwise differences** between subgroup error rates. The headline disparity is:

$$\text{Max Disparity} = \max_s(\text{Error Rate}_s) - \min_s(\text{Error Rate}_s)$$

For Microsoft on PPB, this is $20.8\% - 0.0\% = 20.8$ percentage points (or, as the paper reports the maximum across all three classifiers, $34.7\%$ for IBM on darker females minus $0.0\%$ for Microsoft on lighter males = $34.7$ percentage points, which the paper rounds to "$34.4\%$ difference in error rate between the best and worst classified groups" — the slight discrepancy of 0.3 percentage points may reflect the specific best-group/worst-group pair being IBM darker females vs. Microsoft lighter males, a cross-classifier comparison).

**The error contribution analysis.** Beyond per-subgroup error rates, the paper reports what *proportion* of all errors come from each subgroup. This is computed as:

$$\text{Error Share}_s = \frac{\text{Number of misclassified subjects in subgroup } s}{\text{Total number of misclassified subjects across all subgroups}}$$

The paper reports (Section 4.4) that darker females constitute 21.3% of the PPB benchmark but account for between 61.0% and 72.4% of all classification errors. This is computed by taking the darker female misclassification count (which depends on the classifier — for Microsoft, at a 20.8% error rate on ~270 darker females, roughly 56 misclassified subjects) and dividing by the total misclassified subjects across all subgroups (which, for Microsoft's 6.3% overall error rate on 1,270 subjects, is roughly 80 subjects). The ratio 56/80 ≈ 70%, consistent with the reported range. This analysis is the strongest single-number summary of intersectional disparity: a subgroup constituting one-fifth of the population produces more than three-fifths of the failures.

**The confidence score analysis for IBM.** Because IBM uniquely provides confidence scores, the paper includes a distributional analysis (Figure 4) that is not possible for Microsoft or Face++. The box plots in Figure 4 show the distribution of IBM confidence scores for each of the four subgroups. The key observable pattern: lighter males and lighter females have confidence scores tightly clustered near 1.0 (the maximum confidence), while darker females show a much wider spread, with scores extending down to approximately 0.75 and a median visibly below 1.0.

This analysis serves as a robustness check on the binary classification results. If IBM's classifier were equally confident in its correct and incorrect classifications across subgroups, we might attribute error rate disparities to differences in the "difficulty" of the classification task across subgroups. But the confidence score pattern suggests something different: the classifier is simultaneously *more wrong* and *less certain* on darker females. The low confidence on misclassified darker females indicates the classifier's internal representation is uncertain, not confidently wrong — which in turn suggests that the training data contained insufficient examples of darker females for the model to learn a clear decision boundary. This is a diagnostic finding that points toward training data imbalance as a likely mechanism, even though the paper cannot verify this because the companies do not disclose their training data composition.

**Why this metric structure matters.** The paper's evaluation framework is deliberately designed to be **minimal yet sufficient** for its purpose. It does not propose new fairness metrics, parity constraints, or optimization criteria. It simply applies standard binary classification metrics to intersectional subgroups and reports the results transparently. This is a strategic choice: by using only well-understood metrics (TPR, FPR, PPV, error rate), the paper makes its findings interpretable to a broad audience — including policymakers, journalists, and the general public — without requiring specialized fairness knowledge. The contribution is not in the mathematical sophistication of the metrics but in the **disaggregation strategy** and the **construction of a benchmark that makes disaggregation meaningful**.

---

#### Summary of Design Choices and Their Justifications

**Fitzpatrick scale over race/ethnicity labels**: Phenotype-based labeling avoids within-category variation and geographic instability of racial categories; the scale's clinical validation (dermatology gold standard) provides scientific credibility; aggregating into lighter/darker bins accounts for off-by-one annotation errors in photograph-based labeling.

**Dermatologist-provided definitive labels for PPB**: A single medical expert provides consistency and clinical expertise that non-specialist annotators cannot; this raises the methodological standard for phenotypic benchmarking and reduces the risk that findings are annotation artifacts.

**Binary gender labels imposed by the APIs**: The audit evaluates systems on their own terms using the categories they choose to output; the paper explicitly flags the inadequacy of binary gender while acknowledging it is the operational category.

**Country selection for gender parity and skin-type balance**: Parliamentary sourcing provides administrative ground truth for gender (names, titles), standardized official portraits that control for image quality variation, and public-domain images under non-restrictive licenses.

**Constrained (optimistic) images**: By using standardized portraits with consistent pose, illumination, and expression, the paper isolates skin type and gender as variables and avoids the confound where unconstrained image quality differences could explain disparities. The South Africa subset serves as a within-country control to further rule out between-country imaging confounds.

**Four standard metrics (TPR, FPR, PPV, error rate) per subgroup**: Minimal, interpretable, and comparable to existing NIST evaluation practice; the paper deliberately avoids proposing new fairness metrics to maximize accessibility and auditability.

**Stratified metric computation (no pooling across subgroups)**: Each subgroup is its own evaluation unit; this is what makes intersectional disparities visible — averaging across subgroups would conceal precisely the patterns the paper documents.

**Temporal specificity (April–May 2017)**: Pinpoints the audited API versions for reproducibility and accountability; acknowledges that results are a snapshot that may not generalize to later versions.

**Documentation of API opacity**: Cataloguing what vendors do not disclose (training data, thresholds, benchmark performance) is itself a methodological contribution that establishes the gap between current practice and the paper's proposed transparency standard.

## 4. Key Insights and Innovations

### Innovation 1: Intersectional Subgroup Analysis as a Diagnostic for Hidden Algorithmic Harm

The paper's most fundamental intellectual contribution is not a new metric—it uses standard TPR, FPR, PPV, and error rate—but a **diagnostic strategy**: evaluate classifiers on subgroups defined by the intersection of demographic and phenotypic attributes, rather than on either attribute alone. This seems straightforward in retrospect, but at the time of publication it was essentially absent from computer vision evaluation practice.

Prior work had evaluated gender classification accuracy by gender (Ngan et al., 2015 found female faces were classified 1.8%–12.5% worse than male faces) or by ethnicity proxy (Farinella and Dugelay, 2012, using binary Caucasian/non-Caucasian categories, claimed no ethnicity effect). The dominant implicit assumption was that performance disparities were **univariate**—that you could measure a "gender gap" or a "race gap," and that closing these gaps independently would produce fair systems. The paper demonstrates that this assumption is empirically false and methodologically dangerous.

The evidence is in the numbers. For Microsoft on PPB (Table 4): darker males have a 6.0% error rate, lighter females 1.7%, but darker females 20.8%. A gender-only analysis would report an average female error rate of roughly 10.7% (averaging 20.8% for darker females and 1.7% for lighter females, weighted by subgroup size), which obscures the 20.8% worst case entirely. A skin-type-only analysis would report an average darker-subject error rate of roughly 12.9% (averaging 20.8% for darker females and 6.0% for darker males), again hiding the peak. Neither univariate analysis reveals that one specific subgroup bears a disproportionate share of failures. The error concentration statistic makes this vivid: darker females constitute 21.3% of PPB but produce 61.0%–72.4% of all errors across the three classifiers.

This is not a small refinement of existing fairness methodology—it is a **fundamental reframing** of what it means to audit an algorithm. The reframing has two components: **(1)** the unit of analysis must be the intersectional subgroup, not the single-axis demographic category, because disadvantage can compound multiplicatively across attributes, and **(2)** aggregate accuracy is not just insufficient—it is actively misleading, because it creates an illusion of general competence that survives only as long as you don't ask "competent for whom?" The paper doesn't invent intersectionality (the concept originates in legal scholarship; Crenshaw, 1989), but it operationalizes it as a concrete, replicable evaluation protocol for computer vision systems, transforming an abstract social theory into a measurement apparatus.

The significance extends beyond this specific study's findings. The paper establishes a template: construct a benchmark balanced across intersectional subgroups, compute standard metrics per subgroup, report the full matrix. Any future auditor can adopt this template without specialized fairness expertise—the innovation is the protocol, not the math.

### Innovation 2: Phenotypic Labeling as a Replacement for Race-Based Demographic Auditing in Visual Tasks

The paper makes a deliberate and careful case for why **skin type** (measured via the Fitzpatrick scale) is a more scientifically rigorous and practically useful audit category than race or ethnicity for facial analysis systems. This is a conceptual move, not just a measurement choice, and it challenged the dominant practice in algorithmic fairness auditing at the time.

Prior algorithmic audits—most prominently Angwin et al. (2016) on recidivism prediction—used racial categories (Black, White) as the demographic axis for disparity measurement. For non-visual tasks like risk assessment, this makes sense: the relevant social construct is race, and the harm is disparate treatment or impact along racial lines. But the paper argues that for visual tasks, race labels face two specific failures (Section 3.1): **(1)** wide phenotypic variation within racial categories means a benchmark with lighter-skinned Black subjects cannot reveal failures on darker-skinned Black subjects—the measurement collapses the very variation that matters for visual systems—and **(2)** racial categories are geographically unstable, making them unreliable for international benchmarks like IJB-A that span multiple countries with different racial taxonomies.

The Fitzpatrick scale addresses both. It measures the actual visual property (skin reflectance) that interacts with camera sensors and classifier decision boundaries, rather than a social category imperfectly correlated with that property. The paper further strengthens this argument by citing Roth (2009): "default camera settings are calibrated to expose lighter-skinned individuals," meaning skin type is **causally linked** to input quality through sensor design, not just a demographic correlate. A classifier that fails on darker-skinned subjects may be failing partly because the input images are poorly exposed—a technical problem that race labels alone cannot diagnose, but skin-type labels can.

This is a fundamental shift, not an incremental refinement. It proposes replacing a socially-constructed category (race) with a physically-measured property (skin type) as the primary axis of visual fairness evaluation. The paper is careful not to claim that race-based auditing is unimportant—it explicitly acknowledges that race labels are appropriate for non-visual data—but it argues that for facial analysis, phenotypic labels provide **finer diagnostic resolution** and **greater geographic portability**. The dermatologist validation (a board-certified surgical dermatologist provided definitive labels for PPB) adds scientific credibility, but the deeper contribution is the argument that phenotype, not race, is the right abstraction level for visual system auditing. This has implications beyond gender classification: any facial analysis task (recognition, detection, expression analysis, age estimation) could adopt this labeling protocol, and the paper's call for future work (Section 5) explicitly lists these extensions.

A subtlety worth surfacing: the paper acknowledges that the Fitzpatrick scale itself has a representational bias—Types I, II, and III describe variations of light skin, while Types IV, V, and VI must cover the entire global range from olive to darkest brown. This is the same kind of measurement asymmetry the paper critiques elsewhere. The paper flags this limitation honestly (Section 3.5) but does not resolve it, treating the Fitzpatrick scale as the best available scientifically-validated starting point while implicitly calling for the development of more balanced phenotypic classification systems.

### Innovation 3: The Error Concentration Ratio as an Interpretable Disparity Summary

The paper's third innovation is a specific analytic quantity that makes intersectional harm intuitively graspable without specialized statistical training: the ratio between a subgroup's population share and its share of total errors. The paper reports (Section 4.4) that darker females constitute 21.3% of PPB but account for 61.0%–72.4% of all classification errors across the three evaluated systems. This means the error rate for darker females is not just higher in absolute terms—it is **grossly disproportionate** relative to their representation in the data.

This is not a standard fairness metric. It is not demographic parity, equalized odds, or any of the formal criteria from the fairness literature (Hardt et al., 2016b,a). It is instead a **communication device**: a single sentence that conveys both the magnitude and the distributional injustice of the disparity. "Darker females are 3–3.4× overrepresented among errors relative to their population share" is the underlying mathematical claim, but the paper's formulation ("darker females make up 21.3% of the benchmark but constitute 61.0%–72.4% of the classification error") is more vivid because it makes the asymmetry concrete.

Prior work in algorithmic fairness often reported disparate impact ratios or accuracy gaps, which require some statistical literacy to interpret. The error concentration approach bypasses this by posing a simple counterfactual: if the classifier were equally accurate across all subgroups, each subgroup's share of errors would approximately equal its share of the population. The deviation from this baseline is directly interpretable as a measure of how unequally failure is distributed.

The significance is not mathematical novelty—it's a ratio of two percentages—but **diagnostic clarity**. By collapsing subgroup error rates and subgroup sizes into a single comparison, the paper gives policymakers, journalists, and the public a number they can understand and act on. "The worst-classified group has a 34.7% error rate while the best-classified group has 0.0%" is a statement about absolute performance. "The worst-classified group produces over 70% of all errors despite being one-fifth of the population" is a statement about systemic unfairness. Both are true, but the second carries a normative charge that the first does not. The paper's contribution is recognizing that fairness auditing requires both types of statement and providing the analytic framework to generate both from the same data.

### Innovation 4: The Audit Itself as a Contribution—Commercial Systems Evaluated Without Vendor Cooperation

The paper's fourth innovation is methodological and political rather than technical: it demonstrates that **external auditing of commercial black-box AI systems is possible, necessary, and revealing**, even when vendors disclose essentially nothing about their training data, model architecture, or evaluation procedures. This is not a contribution to classification methodology—it is a contribution to algorithmic accountability infrastructure.

The paper documents (Section 4.2) exactly what the three vendors did not disclose: training data composition, benchmark performance, model architecture, threshold choices, or any subgroup accuracy metrics. Microsoft's documentation vaguely referenced "advanced statistical algorithms" that "may not always be 100% precise." IBM provided confidence scores but no guidance on how to interpret them. Face++'s terms of use explicitly disclaimed any warranties of accuracy. None reported performance on existing gender estimation benchmarks.

Despite this complete opacity, the paper successfully measured per-subgroup error rates, identified the worst-performing subgroup for each classifier, quantified the gap between best and worst groups (up to 34.4 percentage points), and provided diagnostic evidence (IBM's confidence score distributions, Figure 4) pointing toward training data imbalance as a likely mechanism. It did this using only the API outputs (binary labels plus, for IBM, confidence scores) and a carefully constructed benchmark dataset with independently verified ground-truth labels.

This is a fundamental contribution because it shifts the burden of proof. Before this paper, a company could claim its facial analysis system was "highly accurate" based on aggregate metrics and decline to release subgroup performance data, and there was no public evidence to contradict the claim. After this paper, the **absence of subgroup reporting** itself becomes suspicious—if an external auditor with no access to internals can find 34.7% error rates on darker females, a company's refusal to disclose subgroup accuracy implies either negligence (they haven't measured it) or concealment (they have measured it and don't want to share). The paper's explicit definitions of transparency and accountability in Section 5—"providing information on the demographic and phenotypic composition of training and benchmark datasets" and "reporting algorithmic performance on demographic and phenotypic subgroups and actively working to close performance gaps"—operationalize what the vendors should have been doing all along.

The temporal specificity of the audit (April–May 2017) is itself part of the innovation. By timestamping the evaluation, the paper creates a fixed reference point that vendors can be held accountable against. If Microsoft, IBM, or Face++ later improve their systems, the paper's results serve as a baseline; if they do not improve, the persistence of the documented disparities becomes a story of institutional inaction. This is accountability infrastructure: a measurement that makes future progress (or its absence) measurable.

The limitation, which the paper acknowledges implicitly by not claiming otherwise, is that external auditing can measure *that* disparities exist but cannot definitively diagnose *why*. The paper hypothesizes that training data imbalance is the likely cause, based on the IBM confidence score patterns and the general logic of machine learning, but without access to training data or model internals, this remains a hypothesis. The audit reveals the symptom; identifying the underlying disease requires vendor cooperation. The paper's implicit argument is that demonstrating the symptom's severity is a necessary first step toward compelling that cooperation.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluation dataset is the newly constructed Pilot Parliaments Benchmark (PPB), composed of 1,270 unique subjects (parliamentarians from six countries: Rwanda, Senegal, South Africa, Iceland, Finland, Sweden). For preliminary benchmark characterization, the authors also annotated two existing datasets: IJB-A (500 unique subjects, released 2015 by IARPA/NIST) and Adience (2,194 annotatable unique subjects out of 2,284 total, released 2014 as a gender and age classification benchmark). PPB is used for all commercial classifier evaluations; IJB-A and Adience are used only to characterize existing benchmark composition (Section 3.2, 3.5).

- **Base models.** The paper evaluates three commercial gender classification systems, not open-source models: **Microsoft Cognitive Services Face API**, **IBM Watson Visual Recognition**, and **Face++**. These are proprietary, black-box systems whose training data, model architectures, and internal thresholds are not disclosed. Microsoft's service was described as using "advanced statistical algorithms"; IBM and Face++ were described as using "deep learning-based algorithms" (Section 4.2). The systems were accessed through their public APIs in April and May 2017. No model training or fine-tuning was performed by the authors—this is purely an evaluation study of deployed commercial systems.

- **Metrics.** The paper reports four standard binary classification metrics computed independently for each of four intersectional subgroups (darker females, darker males, lighter females, lighter males): **True Positive Rate (TPR)** — the proportion of subjects in a subgroup correctly classified (equivalent to per-subgroup accuracy); **Error Rate** — 1 − TPR, the proportion misclassified; **Positive Predictive Value (PPV)** — the proportion of subjects predicted to belong to a given gender who actually belong to that gender; and **False Positive Rate (FPR)** — the proportion of subjects not belonging to a given gender who are incorrectly classified as that gender (Section 4.3, Table 4). The paper also reports an **error concentration statistic**: the percentage of total classification errors attributable to each subgroup, compared against that subgroup's representation in the benchmark (Section 4.4). For IBM only, confidence score distributions are analyzed (Figure 4). Aggregate accuracy across all subjects is also reported but primarily to demonstrate how it conceals subgroup disparities.

- **Baselines.** This is an audit study, not a comparative methods evaluation. There are no algorithm baselines in the conventional sense. Instead, the "baseline" implicit in the study is the **aggregate accuracy** that each company could report (and that prior benchmarking practice would have accepted as sufficient): 93.7% for Microsoft, 90.0% for Face++, and 87.9% for IBM on PPB (Table 4, "All" column). The paper's contribution is demonstrating that this aggregate baseline is misleading by decomposing it into subgroup-specific metrics. Prior evaluation practice—represented by NIST's gender classification report (Ngan et al., 2015) which reported overall, male, and female accuracy but not skin-type or intersectional breakdowns—serves as the methodological baseline that the paper argues is insufficient.

- **Generation budget / compute accounting.** Not applicable in the conventional sense—the paper does not train models or allocate inference compute budgets. The relevant resource constraint is the **number and diversity of annotated test subjects**. PPB contains 1,270 subjects labeled for both gender and Fitzpatrick skin type (with dermatologist verification), which establishes the statistical power available for subgroup error rate estimation. The smallest subgroup (darker females) contains approximately 270 subjects (21.3% of 1,270), meaning a 20.8% error rate corresponds to roughly 56 misclassifications—sufficient for a reasonably stable estimate. The cost of annotation (three annotators plus a board-certified dermatologist) is briefly described but not quantified in dollar or time terms.

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation or statistical significance testing. This is an observational audit of fixed API endpoints at a single point in time (April–May 2017); there is no model training, no hyperparameter tuning, and no sampling variability beyond what is inherent in the test set. The absence of confidence intervals or hypothesis tests is a limitation—the paper reports raw percentages without characterizing uncertainty, so a 20.8% error rate on darker females and a 1.7% error rate on lighter females are treated as point estimates. The South Africa subset analysis (Table 5) serves as a form of robustness check (replicating the overall pattern within a single country with uniform imaging conditions) rather than a formal statistical validation.

### Main Quantitative Results

The paper organizes its empirical findings into four layers of progressive disaggregation: (1) aggregate accuracy and its deception, (2) single-axis disparities by gender and by skin type, (3) intersectional subgroup error rates that reveal the compound disadvantage of darker females, and (4) the error concentration analysis showing that darker females produce a grossly disproportionate share of all failures. I present these in the order the paper reveals them, since each layer motivates the next.

---

#### Layer 1: Aggregate Accuracy Conceals the Problem

**Headline.** All three commercial classifiers achieve aggregate accuracy between 87.9% and 93.7% on PPB (Table 4, "All" column, TPR row), which by conventional benchmarking standards would suggest these systems are suitable for general deployment.

**Specific numbers from Table 4:**

- **Microsoft**: 93.7% overall TPR (6.3% error rate)
- **Face++**: 90.0% overall TPR (10.0% error rate)
- **IBM**: 87.9% overall TPR (12.1% error rate)

These aggregate numbers are what a vendor might report in marketing materials or API documentation—and indeed, none of the three vendors reported any performance metrics on existing benchmarks (Section 4.2). The paper presents these numbers first to establish that, by the standard the industry had implicitly adopted, these classifiers appear to work well. The subsequent disaggregation demonstrates that the appearance is an artifact of averaging.

---

#### Layer 2: Single-Axis Disparities by Gender and by Skin Type

**Gender disparities (Table 4, F vs. M columns).** All three classifiers perform worse on female subjects than male subjects:

- **Microsoft**: 10.7% error rate for females vs. 2.6% for males — an **8.1 percentage-point gap**.
- **Face++**: 21.3% error rate for females vs. 0.7% for males — a **20.6 percentage-point gap**.
- **IBM**: 20.3% error rate for females vs. 5.6% for males — a **14.7 percentage-point gap**.

The paper notes (Section 4.4) that this replicates NIST's prior finding that "gender classification performance on female faces was 1.8% to 12.5% lower than performance on male faces" (Ngan et al., 2015). But the PPB results show **larger** gaps than NIST reported—particularly for Face++ and IBM, where the female-male error rate differences (20.6 and 14.7 percentage points respectively) substantially exceed the NIST range. The paper does not speculate on why the gaps are larger in its evaluation, but possible factors include: PPB's deliberate inclusion of darker-skinned females (underrepresented in NIST's country-of-origin sampling), differences in the evaluated algorithms (the paper evaluates 2017 commercial APIs; NIST evaluated 2014–2015 algorithms), and differences in benchmark composition.

**Skin-type disparities (Table 4, Darker vs. Lighter columns).** All three classifiers perform worse on darker-skinned subjects than lighter-skinned subjects:

- **Microsoft**: 12.9% error rate for darker subjects vs. 0.7% for lighter subjects — an **12.2 percentage-point gap**.
- **Face++**: 16.5% error rate for darker subjects vs. 4.7% for lighter subjects — an **11.8 percentage-point gap**.
- **IBM**: 22.4% error rate for darker subjects vs. 3.2% for lighter subjects — a **19.2 percentage-point gap**.

The IBM result is particularly striking: the error rate on darker subjects is approximately **7 times** the error rate on lighter subjects (22.4% vs. 3.2%). This is the paper's first demonstration that skin type—independent of gender—is a major axis of performance disparity. However, the paper immediately argues that these single-axis analyses are insufficient, because they cannot reveal whether the female penalty and the darker-skin penalty compound, offset, or interact in more complex ways.

**An important detail in the PPV and FPR columns.** The paper also reports positive predictive value and false positive rate for each subgroup (Table 4). These reveal asymmetric error patterns:

- For Microsoft, the FPR for males (males misclassified as female) is **10.7%**, while the FPR for females (females misclassified as male) is **2.6%**. This means Microsoft's classifier is biased toward predicting "female" when uncertain—it makes more false-female errors than false-male errors.
- For Face++, the asymmetry is reversed and extreme: the FPR for males is **21.3%** while the FPR for females is **0.7%**. Face++'s classifier is heavily biased toward predicting "male," incorrectly classifying over one-fifth of females as male while almost never classifying males as female.
- For IBM, the asymmetry is similar to Face++ but less extreme: male FPR is 20.3%, female FPR is 5.6%.

These asymmetries matter for downstream use. A surveillance system using Face++ to identify "female suspects" would miss 21.3% of actual females (high false negative rate for females) while correctly identifying 99.3% of males. The PPV column shows the flip side: when Face++ predicts "female," it is correct 98.9% of the time (high precision), but this precision comes at the cost of classifying many actual females as male (low recall). This is a classic precision-recall tradeoff, but the thresholds producing it are opaque—set by the vendor without user control (Section 4.6).

---

#### Layer 3: Intersectional Subgroup Error Rates

**Headline.** All three classifiers perform worst on darker-skinned females, with error rates ranging from 20.8% to 34.7%, while lighter-skinned males are classified with near-perfect accuracy (0.0%–0.8% error rates).

**Table 4, DF, DM, LF, LM columns (Error Rate rows):**

| Subgroup | Microsoft | Face++ | IBM |
|---|---|---|---|
| Darker Female (DF) | **20.8%** | **34.5%** | **34.7%** |
| Darker Male (DM) | 6.0% | 0.7% | 12.0% |
| Lighter Female (LF) | 1.7% | 9.8% | 7.1% |
| Lighter Male (LM) | **0.0%** | **0.8%** | **0.3%** |

The pattern is stark and consistent across all three vendors:

**Darker females are the worst-classified group for every classifier.** IBM performs worst on this subgroup (34.7% error rate), meaning more than one-third of darker-skinned female parliamentarians are misgendered by IBM's system. Face++ is nearly as bad (34.5%). Microsoft performs substantially better (20.8%) but still misclassifies one in five darker females.

**Lighter males are the best-classified group for two of three classifiers.** Microsoft achieves a 0.0% error rate on lighter males (perfect classification on this subgroup). IBM achieves 0.3%. Face++ classifies darker males best (0.7% error rate), with lighter males close behind at 0.8%—essentially a tie.

**The gap between best and worst subgroups is enormous.** The paper reports (Section 4.1) a "maximum difference in error rate between the best and worst classified groups" of **34.4%** (which I note is slightly less than the 34.7% IBM darker-female error rate minus 0.0% Microsoft lighter-male error rate = 34.7 percentage points; the 0.3 percentage point discrepancy may reflect a specific pairwise comparison or rounding).

**Within-gender comparisons reveal that skin type amplifies the gender gap.** For females: the darker-female error rate is **12.2× higher** (Microsoft: 20.8% vs. 1.7%), **3.5× higher** (Face++: 34.5% vs. 9.8%), and **4.9× higher** (IBM: 34.7% vs. 7.1%) than the lighter-female error rate. This means the "female penalty" documented in Layer 2 is almost entirely concentrated among darker-skinned females—lighter females are classified relatively well (1.7%–9.8% error rates).

**Within-skin-type comparisons reveal that gender amplifies the skin-type gap for darker subjects but not lighter ones.** For darker subjects: the darker-female error rate is **3.5× higher** (Microsoft), **49.3× higher** (Face++), and **2.9× higher** (IBM) than the darker-male error rate. For lighter subjects, the gender gap is much smaller (Microsoft: 1.7% vs. 0.0%; IBM: 7.1% vs. 0.3%) or reversed (Face++: females 9.8% vs. males 0.8%—still a gap but dwarfed by the 34.5% vs. 0.7% gap for darker subjects).

This is the paper's central empirical finding and its strongest evidence for intersectionality: **the compound disadvantage of being both darker-skinned and female is larger than the sum of the individual disadvantages.** A purely additive model would predict the darker-female error rate as (baseline error) + (female penalty) + (darker penalty). The observed darker-female error rates (20.8%–34.7%) substantially exceed what such a model would predict from the single-axis disparities, meaning there is an interaction effect—the classifier's difficulty with darker skin and its difficulty with female faces multiply rather than add.

---

#### Layer 4: Error Concentration Analysis

**Headline.** Darker females, who constitute 21.3% of the PPB benchmark, account for between 61.0% and 72.4% of all classification errors across the three systems (Section 4.4, and Table 1 in supplementary materials, which the paper references for the exact percentages).

This means that if you took all the misclassifications produced by any of the three classifiers on PPB, roughly two-thirds to three-quarters of them would be darker females—despite darker females being only one-fifth of the test population. The paper frames this vividly (Section 4.4): "Even though darker females make up 21.3% of the PPB benchmark, they constitute between 61.0% to 72.4% of the classification error."

The flip side of this concentration is equally dramatic: **lighter males, who make up 30.3% of the benchmark, contribute only 0.0% to 2.4% of total errors** (Section 4.4, supplementary Table 1). For Microsoft specifically, lighter males contribute 0.0% of errors—they are simply never misclassified—while darker females contribute the entirety of the error that produces a 6.3% aggregate error rate.

---

#### The South Africa Within-Country Analysis

**Headline.** The pattern of intersectional disparity persists when analyzed on the South African subset alone, where imaging conditions are uniform across subjects, confirming that between-country differences in image quality do not explain the results.

**Table 5 reports error rates for the South African subset (n=437, 79.2% darker, 20.8% lighter):**

| Subgroup | Microsoft | Face++ | IBM |
|---|---|---|---|
| Darker Female (DF) | **23.8%** | **36.0%** | **33.1%** |
| Darker Male (DM) | 0.0% | 0.5% | 5.7% |
| Lighter Female (LF) | 0.0% | 7.4% | 0.0% |
| Lighter Male (LM) | 0.0% | 0.0% | 1.6% |

The pattern is essentially identical to the full PPB results:

- Microsoft: all error (23.8%) is concentrated in darker females; darker males and all lighter subjects have 0.0% error. The paper states (Section 4.4): "all the error for Microsoft arises from misclassifying images of darker females."
- Face++: darker females have a 36.0% error rate, vastly exceeding all other subgroups (0.0%–7.4%).
- IBM: darker females have a 33.1% error rate, followed by darker males at 5.7% and lighter subjects near 0.0%.

The paper's interpretation (Section 4.7) is that "variation in performance due to the image characteristics of each country does not fully account for the differences in misclassification rates between intersectional subgroups." Because South African parliamentary images have consistent pose and illumination (the paper describes them as having "the most consistent pose and illumination" among all parliaments) and contain substantial numbers of both lighter and darker subjects, the persistence of the disparity within this subset rules out the hypothesis that the gap is simply an artifact of lower-quality photography in African parliaments.

**A nuance the paper adds**: "darker skin alone may not be fully responsible for misclassification. Instead, darker skin may be highly correlated with facial geometries or gender display norms that were less represented in the training data of the evaluated classifiers." This is a measured causal caveat—the paper can demonstrate *that* darker females are more misclassified, but it cannot isolate *whether* the mechanism is skin reflectance (affecting image exposure), facial feature distributions differing across populations (affecting classifier decision boundaries), or gender presentation norms differing across cultures (affecting what the classifier learns to associate with "female"). All three mechanisms predict the observed pattern, and the paper's data cannot distinguish them.

---

#### IBM Confidence Score Analysis

**Headline (Figure 4).** IBM's confidence scores are tightly clustered near 1.0 for lighter males and lighter females, but are substantially lower and more variable for darker females, indicating that the classifier is simultaneously more error-prone and less certain on this subgroup.

Figure 4 displays box plots of IBM confidence scores for the four intersectional subgroups. The key observable patterns:

- **Lighter males**: Confidence scores are compressed near 1.0 with minimal variance. The classifier is consistently highly confident—and, per Table 4, almost always correct (0.3% error rate).
- **Lighter females**: Similarly compressed near 1.0 with slightly more spread than lighter males. The classifier is confident and relatively accurate (7.1% error rate).
- **Darker males**: Scores show more spread than lighter subjects but the median remains near 1.0. The classifier is moderately confident and moderately accurate (12.0% error rate).
- **Darker females**: Scores show the widest spread, extending down to approximately 0.75, with a median visibly below 1.0. The classifier is least confident—and, per Table 4, least accurate (34.7% error rate).

The diagnostic value of this analysis is that it distinguishes between two types of classifier failure: **confident errors** (the classifier is wrong but certain, suggesting it has learned a systematically incorrect decision boundary) versus **uncertain errors** (the classifier is wrong and knows it is uncertain, suggesting insufficient training data in that region of feature space). The IBM results are consistent with the latter—the classifier's low confidence on darker females suggests its internal representation lacks sufficient examples of darker female faces to form a clear decision boundary. This is evidence for the training-data-imbalance hypothesis, though not dispositive without access to the actual training data.

The paper also notes (Section 4.6) that IBM was the only vendor to provide confidence scores, and that this additional information is valuable for auditing: "While confidence values give users more information, commercial classifiers should provide additional metrics." The fact that Microsoft and Face++ do not provide confidence scores means external auditors cannot perform this diagnostic analysis on those systems, further limiting accountability.

---

### Ablation Studies and Robustness Checks

Given that this is an audit study rather than a model development paper, the "ablations" are different in character—they are robustness checks on the validity of the measured disparities rather than controlled experiments on model components. I organize them by the potential confound they address.

**Skin-type aggregation (lighter/darker binary vs. six Fitzpatrick types).** The paper primarily reports results with skin types aggregated into lighter (I–III) and darker (IV–VI) bins, but provides the full six-type disaggregation in supplementary Table 2. The aggregation is justified as a hedge against off-by-one annotation errors (Section 3.5): "The skin types are aggregated to account for potential off-by-one errors since the skin type is estimated using images instead of employing a standard spectrophotometer and Fitzpatrick questionnaire." The availability of the full six-type breakdown allows readers to verify that the aggregation does not conceal important within-category variation (e.g., whether Type VI subjects are more misclassified than Type IV subjects within the "darker" category).

**South Africa as within-country control (Table 5).** As detailed in the main results section above, the replication of the intersectional disparity pattern on the South African subset (n=437), where imaging conditions are uniform, rules out the confound that between-country differences in photographic quality drive the observed disparities. This is the paper's strongest robustness check because it directly addresses the most obvious alternative explanation: that African parliamentary photos are simply lower quality than European ones. The South Africa analysis shows that even within a single country with consistent image characteristics, darker females are misclassified at dramatically higher rates than lighter males.

**Per-Fitzpatrick-type analysis (supplementary Table 2).** The paper reports gender classification performance disaggregated by each of the six Fitzpatrick skin types individually. This analysis allows readers to assess whether the lighter/darker binary aggregation masks important variation—for example, whether the "darker female" penalty is concentrated in Type VI subjects specifically or distributed across Types IV, V, and VI. The paper does not discuss this table in detail in the main text, which is a minor limitation—a reader relying solely on the main body would not know whether the intersectional disparity is uniform across darker skin types or concentrated at the extremes.

**IBM confidence score threshold robustness (Figure 4).** By reporting the full distribution of confidence scores rather than just the binary classification, the paper implicitly tests whether the observed disparities are artifacts of IBM's chosen classification threshold. If IBM raised or lowered its internal threshold, the binary error rates would change (with corresponding changes in TPR and FPR), but the confidence score distributions would remain the same. The fact that darker females have lower *confidence* even when correctly classified suggests that the disparity is not purely a threshold artifact—it reflects genuine differences in classifier separability across subgroups. However, because Microsoft and Face++ do not provide confidence scores, this analysis is limited to IBM and cannot be generalized to the other two vendors.

**Multiple annotators for PPB gender and skin-type labels (Section 3.4).** The paper used three annotators for PPB labels, with a board-certified surgical dermatologist providing definitive Fitzpatrick labels. This reduces the risk that annotation errors drive the results—a single mislabeled subject in a small subgroup could substantially shift the error rate estimate. The use of administrative ground truth for gender (parliamentary records listing names, titles, and prefixes like Mr./Ms.) further reduces annotation error for the gender axis. The paper does not report inter-annotator agreement statistics (e.g., Cohen's kappa), which would have strengthened confidence in the label reliability, but the dermatologist's involvement is a strong signal of annotation quality.

**Coverage of the African diaspora.** By including parliamentarians from Rwanda, Senegal, and South Africa, the paper samples from East Africa, West Africa, and Southern Africa respectively. These regions have distinct population histories and phenotypic distributions, and the paper's inclusion of multiple African countries reduces the risk that results are specific to one population. The fact that dark-skinned females from all three African countries are misclassified (though the paper does not report per-country error rates for African nations separately, making it impossible to verify this directly from the reported data) supports the generalizability of the finding across African populations.

### Critical Assessment

#### Do the Experiments Support the Claim That Aggregate Metrics Conceal Subgroup Failures?

**Yes, directly and conclusively.** The paper's central claim is that reporting only aggregate accuracy hides severe subgroup performance disparities. The evidence for this is straightforward and overwhelming: Table 4 shows aggregate accuracy between 87.9% and 93.7% alongside subgroup error rates as high as 34.7% for darker females. The gap between what an aggregate metric suggests ("this system works well for everyone") and what subgroup metrics reveal ("this system fails on one-third of darker females") is the paper's core empirical finding, and the data support it without requiring complex statistical inference. The error concentration statistic—darker females producing 61.0%–72.4% of all errors despite being 21.3% of the benchmark—further reinforces the point by showing not just that a subgroup has higher error rates, but that the system's failures are overwhelmingly concentrated in that subgroup.

This claim is not conditional on model choice, dataset composition, or evaluation protocol—it is a logical consequence of the interaction between any imbalanced dataset and any classifier with subgroup-dependent accuracy. The paper could have made this point with one classifier and one dataset; having three classifiers and a purpose-built balanced benchmark makes the demonstration more compelling, but the core logic is robust.

#### Do the Experiments Support the Claim That Intersectional Analysis Reveals Disparities Invisible to Single-Axis Analysis?

**Yes, with precise quantitative evidence.** The paper claims that evaluating by gender alone or skin type alone misses the compound disadvantage of darker females. The evidence is in the comparison between single-axis error rates and intersectional error rates (all from Table 4):

- Gender alone: female error rates are 10.7% (Microsoft), 21.3% (Face++), 20.3% (IBM). These numbers mask the fact that lighter females have error rates of 1.7%, 9.8%, and 7.1% respectively, while darker females have error rates of 20.8%, 34.5%, and 34.7%. A gender-only analysis would conclude that "females have 10.7% error" and miss that this average conceals a massive split by skin type.
- Skin type alone: darker-subject error rates are 12.9% (Microsoft), 16.5% (Face++), 22.4% (IBM). These numbers mask the fact that darker males are relatively well-classified (6.0%, 0.7%, 12.0%) while darker females are heavily misclassified (20.8%, 34.5%, 34.7%).

The paper quantifies this directly: for Microsoft, the 10.7% aggregate female error rate is the weighted average of 20.8% (darker females) and 1.7% (lighter females). The single-axis number collapses a 19.1 percentage-point gap between female subgroups into a single summary, which is precisely the form of concealment the paper critiques. The evidence for this claim is strong and directly presented.

#### Do the Experiments Support the Claim That Darker-Skinned Females Are the Most Misclassified Group?

**Yes, across all three classifiers and the South Africa subset.** Table 4 shows darker females have the highest error rate for Microsoft (20.8%), Face++ (34.5%), and IBM (34.7%). Table 5 replicates this on the South Africa subset: darker female error rates of 23.8% (Microsoft), 36.0% (Face++), 33.1% (IBM). There is no classifier and no evaluation subset where darker females are not the worst-classified group. The consistency across vendors—with different training data, different model architectures, and different development contexts (US-based Microsoft and IBM, China-based Face++)—suggests this is a systemic problem in facial analysis technology, not an idiosyncratic failure of one vendor.

A caveat: the paper does not report error rates by Fitzpatrick type within the "darker" category in the main text, so a reader cannot determine whether the disparity is driven disproportionately by Type VI subjects (the darkest skin) versus Type IV or V subjects. The supplementary Table 2 provides this breakdown, but the main text's decision to aggregate into lighter/darker bins means the headline finding is coarser than it could be. If the disparity were concentrated entirely in Type VI subjects—who may have been extremely rare in the training data of these classifiers—the policy implications would be different than if the disparity is graded across Types IV, V, and VI. The paper's decision to report the aggregated results in the main text while providing the granular breakdown in supplementary materials is a reasonable presentation choice, but the main-text analysis is less informative than it could be.

#### Do the Experiments Support the Claim of a 34.4 Percentage-Point Gap Between Best and Worst Groups?

**Yes, but this is a cross-classifier comparison, not a within-classifier comparison.** The paper reports (Section 4.1) that "the maximum difference in error rate between the best and worst classified groups is 34.4%." This figure compares IBM's error rate on darker females (34.7%) against Microsoft's error rate on lighter males (0.0%)—or possibly Face++'s error rate on darker males (0.7%) against IBM's error rate on darker females (34.7%), depending on the exact pairing. Either way, this is a comparison across different classifiers, not the gap within a single classifier. The within-classifier gaps are: Microsoft 20.8% (DF 20.8% vs. LM 0.0%), Face++ 33.8% (DF 34.5% vs. DM 0.7%), IBM 34.4% (DF 34.7% vs. LM 0.3%). The 34.4% headline number corresponds to IBM's within-classifier gap.

The cross-classifier comparison is less meaningful as a measure of any one system's fairness—it tells you that the best-performing system on the easiest subgroup does much better than the worst-performing system on the hardest subgroup, which is not surprising. The within-classifier gaps (20.8–34.4 percentage points) are more interpretable as measures of each system's internal disparity. The paper would have been clearer to lead with the within-classifier gaps rather than maximizing the number by crossing classifiers.

#### Genuine Weaknesses in the Experimental Design

**Single benchmark, single task.** All evaluations are performed on PPB, a dataset of constrained parliamentary portraits, for the single task of binary gender classification. The paper does not evaluate on unconstrained ("in-the-wild") images, does not evaluate other facial analysis tasks (face detection, face recognition, age estimation, expression recognition), and does not evaluate on other demographic or phenotypic axes beyond binary gender and Fitzpatrick skin type. The paper explicitly calls for extensions to other tasks (Section 5), but the current results are specific to gender classification on high-quality official portraits. It is plausible—even likely—that disparities would be larger on unconstrained images with pose, illumination, and expression variation, where the sensor biases the paper discusses (camera exposure optimized for lighter skin) would interact with other image quality degradations. But this hypothesis is untested.

**Small sample sizes for some subgroups in the granular analysis.** The per-Fitzpatrick-type breakdown (supplementary Table 2) splits the 1,270 subjects into 12 cells (6 skin types × 2 genders), some of which are very small. For example, Type VI darker females likely constitute a small fraction of the 21.3% darker-female share (since Types IV, V, and VI share that 21.3%). If Type VI females number only a few dozen subjects, the per-cell error rate estimates are noisy, and the paper does not report confidence intervals to convey this uncertainty. The decision to aggregate into four intersectional groups rather than twelve is statistically prudent, but it means the paper cannot speak to whether the disparity is graded across darker skin types or concentrated at the extreme.

**No statistical significance testing or confidence intervals.** The paper reports all results as point estimates without standard errors, confidence intervals, or hypothesis tests. For example, "20.8% error rate" on ~270 darker females for Microsoft corresponds to approximately 56 misclassifications. The 95% binomial confidence interval for this proportion is roughly 16.0%–26.2% (using the normal approximation, which is reasonable for n=270 with p=0.208). This means the darker-female error rate could plausibly be as low as 16% or as high as 26%. Meanwhile, the 1.7% error rate on lighter females (~296 subjects, ~5 misclassifications) has a confidence interval of roughly 0.6%–3.9%. The difference between these two rates is clearly statistically significant (the confidence intervals don't overlap), but the paper does not quantify this, leaving readers to assess reliability by intuition. For a paper aimed at establishing auditing standards, this omission is notable—future audits following this template should include uncertainty quantification.

**Temporal snapshot without replication.** The paper evaluates each API at a single point in time (April–May 2017). Commercial APIs are continuously updated, and the paper cannot claim that the measured disparities persisted beyond the evaluation window. The paper acknowledges this implicitly by timestamping the evaluation, but it does not discuss whether the vendors might have updated their systems during the evaluation period itself (e.g., if IBM deployed a new model in mid-May after the paper had already collected partial results). The lack of temporal replication is a limitation on generalizability but is inherent to the audit methodology—external auditors cannot freeze vendor systems. The paper's response is to document the evaluation timeframe precisely, which is the best available practice under the circumstances.

**No verification that API outputs are deterministic.** The paper does not report whether submitting the same image to the same API multiple times produces the same gender classification. If the APIs include stochastic elements (e.g., random data augmentation at inference time, or load-balancing across different model versions), then the reported error rates are averages over unknown stochastic variation. For IBM, the confidence scores would partly capture this (a subject near the decision boundary might receive different binary classifications on different API calls), but for Microsoft and Face++, which return only binary labels, there is no way to assess output stability. The paper implicitly assumes deterministic outputs, which may be reasonable for commercial APIs at the time but is not verified.

#### Missing Experiments That Would Have Strengthened the Paper

**Unconstrained image evaluation.** The paper characterizes PPB as "highly constrained" and "an optimistic sample" (Section 4.7). Evaluating the same classifiers on an unconstrained benchmark with similar phenotypic diversity—such as a balanced subset of images with pose, illumination, and expression variation—would test whether the disparities are larger under realistic deployment conditions. The paper's hypothesis, stated in Section 4.7, is that "error rates would be higher on more challenging unconstrained datasets," but this prediction is not tested.

**Face detection pipeline evaluation.** The paper evaluates gender classification as a standalone task. In real deployments, gender classification typically follows face detection—the system must first locate the face in an image before classifying its gender. If face detection also exhibits skin-type and gender disparities (which the paper hypothesizes based on the benchmark-construction feedback loop discussed in Section 2), then the end-to-end pipeline error rates could be substantially different from the gender-classification-only error rates reported here. Evaluating face detection + gender classification jointly would provide a more ecologically valid assessment.

**Inter-annotator agreement for skin-type labeling.** The paper uses three annotators plus a dermatologist for PPB labels but does not report any measure of annotator agreement (e.g., Fleiss' kappa, Krippendorff's alpha, or simple pairwise agreement rates). This makes it impossible to assess the reliability of the skin-type labels independently. If the three annotators frequently disagreed before the dermatologist resolved the labels, the Fitzpatrick categories may be less reliable as an audit instrument than the paper implies. If they consistently agreed, that strengthens the methodology. The paper misses an opportunity to establish the measurement reliability of its primary phenotypic instrument.

**Evaluation on a darker-skinned-only benchmark.** Constructing a benchmark composed entirely of darker-skinned subjects (say, 500 parliamentarians from multiple African countries) and evaluating all three classifiers on it would provide a direct test of whether the classifiers' aggregate accuracy degrades as the proportion of darker subjects increases. This would complement the PPB's balanced design with an extreme-case stress test.

**Longitudinal audit.** Re-evaluating the same APIs at multiple time points (e.g., every 3 months for a year) would reveal whether vendors respond to public pressure by improving subgroup performance. The paper does not attempt this, but its timestamped methodology enables future researchers to perform exactly this comparison.

#### Conditional Nature of the Claims

The paper's findings are conditional on the following scope limitations, which it mostly acknowledges:

**Task scope.** The findings apply to binary gender classification from facial images. They do not necessarily generalize to face recognition, face detection, age estimation, emotion recognition, or other facial analysis tasks—though the paper hypothesizes they will and calls for replication. The causal mechanism (biased training data leading to worse performance on underrepresented phenotypes) is general, but the magnitude and subgroup pattern may differ across tasks.

**Benchmark characteristics.** The findings are demonstrated on constrained, high-quality parliamentary portraits with cooperative subjects, consistent illumination, and limited pose variation. The paper explicitly notes that this is an "optimistic" evaluation and that unconstrained images would likely show larger disparities, but this remains a hypothesis.

**Phenotypic axis.** The findings use Fitzpatrick skin type as the phenotypic measure. Other phenotypic attributes—facial geometry, hair texture, nose shape—are not measured, and the paper acknowledges that "darker skin may be highly correlated with facial geometries or gender display norms" that are the actual causal factors (Section 4.7). The paper measures skin type accurately but cannot isolate whether skin type is the cause or a correlate of the observed disparities.

**Temporal specificity.** The findings represent the state of Microsoft, IBM, and Face++ APIs in April–May 2017. They should not be assumed to describe the same APIs at later dates, nor other vendors' systems, without replication.

**Vendor population.** The findings apply to three major commercial vendors who provided public APIs at the time. They do not represent all facial analysis systems—open-source models, academic research systems, or government-developed systems might exhibit different disparity patterns. The paper's selection of vendors is well-justified (market share, public availability, geographic diversity) but cannot claim representativeness of the entire industry.

**Binary gender limitation.** The findings use binary gender labels imposed by the APIs. The paper cannot speak to how these systems would perform on non-binary, transgender, or gender-non-conforming individuals, because the APIs do not offer those classification categories and PPB does not include those labels. This is a genuine limitation that the paper flags (Section 3.4) but cannot address with its methodology.

## 6. Limitations and Trade-offs

### Limitation 1: Difficulty Estimation Cost Is Unaccounted for and Dominates the Inference Budget

**The assumption or constraint.** The entire compute-optimal framework rests on knowing each prompt's difficulty *before* deciding how to allocate the test-time compute budget. The paper's difficulty estimation method generates 2,048 samples per question and averages either ground-truth correctness (oracle) or PRM final-answer scores (predicted) to assign questions to difficulty quintiles (Section 3.2). The paper explicitly acknowledges this cost is not included in the efficiency calculations: "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity" (Section 3.2).

**The consequence.** The paper's headline finding—that compute-optimal scaling achieves "more than 4× better efficiency" over best-of-N—is computed *after* difficulty is known, excluding the cost of learning it. In practice, generating 2,048 samples to estimate difficulty consumes more compute than the largest test-time budgets studied (256–512 generations). If difficulty estimation were amortized into the total budget, the efficiency gains over best-of-N would shrink substantially or disappear entirely. The paper frames this as an "exploration-exploitation tradeoff" (Section 3.2), meaning some of the test-time budget must be sacrificed to assess difficulty before solving the problem. The 4× figure should therefore be understood as an **upper bound** on achievable efficiency in deployment, not a realized gain.

**What evidence exists in the paper.** The paper presents the difficulty bins as a given input to the compute-optimal policy (Figures 4, 8) and never integrates the cost of producing those bins into the efficiency analysis. The predicted-difficulty variant eliminates the need for ground-truth labels but still requires generating 2,048 samples per prompt and scoring them with the PRM—it reduces the *information* requirement (no answers needed) but not the *computational* requirement. The paper acknowledges this explicitly (Section 3.2) but provides no experimental results measuring the tradeoff.

**Mitigation status.** The paper flags this as "a key area for future work" (Section 3.2) and suggests training models to predict difficulty directly from question text, eliminating the per-prompt sampling overhead. It does not develop or evaluate any such method. Until lightweight difficulty estimation is demonstrated, the compute-optimal framework is impractical for deployment settings where each prompt is encountered once and must be solved within a fixed budget.

---

### Limitation 2: Hard Problems Remain Fundamentally Unsolved—Test-Time Compute Cannot Create Capability

**The assumption or constraint.** The compute-optimal framework assumes the base model already produces correct solutions at some non-trivial rate for the prompts being processed. The paper finds that on the hardest questions (difficulty bin 5), the base model's pass@1 is near zero, and no allocation strategy—search, revisions, or their compute-optimal combination—makes meaningful progress regardless of budget (Section 7). The paper is transparent about this boundary: "test-time compute amplifies existing capability but does not create it from nothing."

**The consequence.** For any problem class where the base model's pass@1 is effectively zero, the entire framework offers no benefit over any other strategy—all methods produce uniformly poor results. This is not a minor edge case: on the MATH benchmark, the hardest quintile of questions (bin 5) shows accuracy near 1–3% for all methods, all budgets, and all allocation strategies (Figures 3 right, 7 right, 9). The FLOPs-matched comparison (Section 7) confirms that on hard questions, pretraining a larger model is almost always preferable to investing in test-time compute—at R >> 1, PRM search shows a -52.9% relative disadvantage compared to the ~14× larger model (Figure 1, bottom-right bar chart). This means the framework provides no guidance for genuinely novel or out-of-distribution reasoning that exceeds the base model's training distribution. For such problems, pretraining remains the only viable path.

**What evidence exists in the paper.** The difficulty-bin analyses in Figures 3 (right), 7 (right), and 9 consistently show bin 5 performance as nearly flat and near zero across all methods and budgets. In Figure 3 (right), beam search and best-of-N both hover at 1–3% accuracy on bin 5 across all budget levels from 4 to 256 generations. In Figure 7 (right), all sequential-to-parallel ratios produce roughly 2–3% accuracy on bin 5. In Figure 9, the bin 5 scaling line is flat and near 0–5% for both revisions and PRM search.

**Mitigation status.** The paper acknowledges this limitation explicitly in the Section 7 takeaway box: "test-time and pretraining compute are not 1-to-1 exchangeable... test-time compute cannot compensate for fundamental capability gaps." However, the paper does not provide guidance on *how to identify* whether a given problem falls into this "unsolvable" regime without generating 2,048 samples per question, which itself consumes the budget. A practical deployment would need a cheap method to recognize when test-time compute is futile and either escalate to a larger model or flag for human review.

---

### Limitation 3: Single Benchmark, Single Model Family, Single Task—Unknown Generalizability

**The assumption or constraint.** All experiments use the MATH benchmark (500 test questions, high-school competition math) with PaLM 2-S* as the base model (Section 4). The authors state they "believe this model is representative of the capabilities of many contemporary LLMs," but provide no evidence beyond this assertion. The paper does not evaluate on any other reasoning domain (code generation, logical reasoning, scientific QA), any other model family (GPT, LLaMA, Claude), or any task that requires factual recall rather than multi-step inference.

**The consequence.** Five aspects of the findings could be model-specific or domain-specific, and the paper provides no evidence to assess this:

- **PRM quality and over-optimization behavior** depend on the base model's output distribution and the Monte Carlo rollout training procedure. A different base model with different calibration or different error patterns might exhibit qualitatively different difficulty-dependent scaling curves—for example, a model with more uniform pass@1 across difficulty might show less extreme differences between easy and hard problems.
- **The revision model's ability to learn from incorrect in-context examples** depends on the base model's in-context learning capabilities, which vary substantially across model families.
- **The optimal sequential-to-parallel ratio** found for MATH problems (easy → fully sequential, hard → balanced) may not generalize to tasks with different structure—code generation, for instance, might benefit from parallel exploration even on easier problems because there are many valid solutions.
- **The FLOPs-matched comparison** uses PaLM 2-S* vs. a ~14× larger PaLM 2 model with parameter-only scaling (not Chinchilla-optimal). The finding that test-time compute can outperform a larger model on easy-to-medium problems may not hold for model families with different scaling properties.
- **The MATH benchmark** consists of problems with unambiguous correct answers that can be verified with string matching. The PRM training pipeline (Monte Carlo rollouts) and difficulty estimation (pass@1) both depend on this clean correctness signal. Extending to open-ended generation or tasks with ambiguous correctness would require fundamentally different verifier and difficulty estimation approaches.

**What evidence exists in the paper.** The paper provides no cross-domain, cross-model, or cross-task evaluations. All figures, tables, and claims are based on PaLM 2-S* on MATH. This is a deliberate scope choice—the paper's contribution is a framework and an initial demonstration—but it leaves the generality of the findings entirely untested.

**Mitigation status.** The paper does not claim generality beyond MATH and PaLM 2-S*, and the model is described as "representative" rather than "universal." The paper does not attempt replication on other benchmarks or model families. Section 8 acknowledges this implicitly by calling for future work on applying the framework to other tasks, but the lack of any multi-domain validation is a significant gap for practitioners deciding whether to adopt the approach in their own domain.

---

### Limitation 4: Revisions and Search Are Studied Independently—No Combined System

**The assumption or constraint.** The paper studies two complementary mechanisms—PRM-guided search (modifying the verifier, Section 5) and iterative revisions (modifying the proposal distribution, Section 6)—but never combines them into a single system. Section 8 explicitly states: "we did not experiment with PRM tree-search techniques in combination with revisions."

**The consequence.** The paper cannot answer the most natural follow-up question: *would combining PRM search with the revision model as the proposal distribution yield gains beyond either method alone?* The two mechanisms have complementary strengths: revisions improve the quality of generated candidates (better initial solutions), while PRM search improves candidate selection (better discrimination among candidates). Applying beam search to revision model outputs—or using the PRM to guide *which* revision to pursue at each step—could break through the performance ceilings each method individually hits. The current results therefore represent a **lower bound** on what a fully integrated system could achieve, and a practitioner building a production system would want to combine them—but the paper provides no guidance on how to do so or whether the gains are additive, synergistic, or partially redundant.

**What evidence exists in the paper.** The paper documents each mechanism's strengths independently: search helps on medium-difficulty problems (Figure 3, right), revisions help on easy problems (Figure 7, right). The conceptual framework in Section 2 explicitly frames these as complementary axes (proposal distribution vs. verifier), and Figure 5 illustrates how they could be combined, but the paper runs no experiments evaluating the combination. The ReST$^{EM}$ experiment (Appendix K, Figure 16) shows that optimizing the revision model with RL-style training degraded performance—a cautionary signal that naïvely combining search and revisions could backfire, but the paper does not explore whether a different integration strategy would succeed.

**Mitigation status.** The paper acknowledges this gap in Section 8 and identifies it as a key direction for future work. However, the omission is not a minor ablation—it is the central synthesis that the conceptual framework promises but does not deliver. A reader who accepts the paper's argument that proposal and verifier improvements are complementary would naturally expect the paper to demonstrate this complementarity empirically, and the absence of that demonstration is a significant gap.

---

### Limitation 5: The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate and Revision Training Is Fragile

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect followed by a correct target (Section 6.1). It never sees examples where the current answer is already correct and should be preserved. The paper reports that "approximately 38% of correct answers get converted back to incorrect ones" during sequential revision.

**The consequence.** The revision chain is not monotonically improving—at each step, the model may take a currently correct answer and "revise" it into an incorrect one. This creates a fundamental reliability problem: a practitioner cannot simply run N sequential revisions and take the final answer. The paper mitigates this with within-chain selection (majority voting or verifier-based selection across all steps in the chain), but these are patches that add computational overhead (the verifier must evaluate every intermediate answer) and still discard useful information (the model's own implicit assessment of when revision is needed). More fundamentally, the 38% reversion rate means that purely sequential strategies leave a substantial fraction of "solvable" problems unsolved because the model actively destroys correct answers it previously generated.

**What evidence exists in the paper.** The paper reports the 38% figure directly (Section 6.1) and documents the within-chain selection mitigation (Figure 5, Appendix I). The ReST$^{EM}$ experiment (Appendix K, Figure 16) provides further evidence of fragility: attempting to optimize the revision model with on-policy RL-style training caused sequential revision performance to drop from ~38.5% at the optimal sequential-to-parallel ratio to ~33.5% at fully sequential at 256 generations. This suggests the positive revision results depend on specific training choices (offline data construction, edit-distance-based incorrect-to-correct pairing) that may not transfer to other training paradigms.

**Mitigation status.** The paper mitigates the reversion problem with verifier-based or majority-based selection across the entire revision chain (Section 6.1, Appendix I), which reduces the impact of reversions on final accuracy but does not eliminate it—the selected answer might be a correct earlier answer that was subsequently revised into an error, but selection is imperfect. The paper does not explore more principled solutions, such as training the revision model to output a "no revision needed" token when the current answer is correct, or using the PRM's step-level scores to decide whether to accept or reject each revision. The fragility exposed by the ReST$^{EM}$ experiment is noted but not explained—the paper hypothesizes that "on-policy data collection exacerbates spurious correlations" (Appendix K) but does not investigate further.

---

### Limitation 6: No Accounting for Latency—Sequential Strategies Increase Wall-Clock Time

**The assumption or constraint.** The paper measures test-time compute in "generations" (number of complete solutions sampled), which is a proxy for total FLOPs but ignores the **serial dependency structure** of different strategies. Sequential revisions are inherently serial—each revision must complete before the next begins—while parallel best-of-N can execute all N samples simultaneously given sufficient hardware.

**The consequence.** A strategy that allocates 128 generations as 64 sequential revisions × 2 parallel chains (a configuration the compute-optimal policy might select for hard problems per Figure 7) takes approximately 64× longer wall-clock time than 128 parallel samples run simultaneously. For latency-sensitive applications—interactive assistants, real-time decision-making, or any deployment where users wait for responses—the sequential-heavy strategies favored by the compute-optimal policy on easy-to-medium problems may be impractical regardless of their accuracy advantages. The paper's FLOPs efficiency gains (4× fewer generations for equivalent accuracy) could translate to *worse* wall-clock performance if the efficient strategy is predominantly sequential.

**What evidence exists in the paper.** The paper provides no latency analysis, no wall-clock time measurements, and no discussion of the throughput implications of sequential vs. parallel strategies. Figure 5 illustrates the sequential and parallel architectures side by side, implicitly acknowledging they have different temporal structures, but the efficiency analysis in Figures 4 and 8 treats all generations as equivalent regardless of whether they are executed serially or in parallel.

**Mitigation status.** Not addressed. The paper's compute model (Section 5.3) treats all generations as having equal cost regardless of serial dependency. A latency-aware extension of the compute-optimal framework—incorporating a wall-clock budget alongside a FLOPs budget—would be a natural extension, but the paper does not propose it. For practitioners, the latency implications of the recommended strategies must be assessed independently based on their hardware parallelism and latency requirements.
