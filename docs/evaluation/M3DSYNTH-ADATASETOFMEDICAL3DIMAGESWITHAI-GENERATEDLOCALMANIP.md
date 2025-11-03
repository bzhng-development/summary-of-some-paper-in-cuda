# M3DSYNTH: A DATASET OF MEDICAL 3D IMAGES WITH AI-GENERATED LOCAL MANIPULATIONS

**ArXiv:** [2309.07973](https://arxiv.org/abs/2309.07973)

## 🎯 Pitch

M3Dsynth introduces the first large-scale dataset of over 8,500 lung CT scans with precisely controlled AI-generated manipulations, such as injection or removal of cancer nodules, enabling robust training and benchmarking of medical image forensic detectors. This resource addresses a critical gap in the fight against medical deepfakes, demonstrating that current automated diagnosis tools can be easily fooled—but that with M3Dsynth, forensic methods can accurately detect and localize tampered regions even when forgeries originate from new, unseen generative models, helping safeguard clinical integrity and patient safety.

---

## 1. Executive Summary
This paper introduces M3Dsynth, a large-scale dataset of 8,577 locally manipulated lung CT scans designed to study and benchmark detection of AI-generated tampering in 3D medical images. It demonstrates that such manipulations can mislead automated cancer classifiers and shows that modern forensic models, once fine-tuned on M3Dsynth, can accurately detect and localize forgeries—even when the training and test manipulations come from different generative methods.

## 2. Context and Motivation
- Problem addressed
  - The paper targets the detection and localization of local AI-based tampering in 3D medical images (CT scans), specifically the injection, enlargement, removal, or shrinkage of lung cancer nodules (Section 1).
  - Prior work shows that attackers could infiltrate hospital systems and alter CT scans to add or remove nodules, deceiving both automated tools and experts (cited experiment in [1]). Yet, comprehensive datasets to develop robust detectors are lacking.

- Why it matters
  - Real-world impact: Manipulated CT scans can cause misdiagnosis, enable insurance fraud, and damage scientific integrity or public safety (Section 1).
  - Research significance: Forensics for deepfakes has focused on faces and natural images; medical CT scans differ fundamentally (no camera pipeline artifacts, often uncompressed), so detectors trained on general imagery do not transfer well (Section 3).

- Shortcomings of prior approaches
  - Existing medical-deepfake studies often use small datasets (e.g., ~100 samples in [1]) and evaluate detection only on manipulations seen during training, leaving cross-generator generalization largely unexplored (Sections 1–2).
  - General-purpose synthetic-image detectors trained on natural images fail on medical CTs: Table 2 (top) shows ~50% accuracy—equivalent to random guessing—when tested on M3Dsynth after training on images from common generative models (ProGAN, StyleGAN2, LDM).

- Positioning
  - M3Dsynth fills the data gap with 8,577 3D manipulated samples and a transparent generation pipeline (Fig. 2; Table 1).
  - The dataset enables rigorous evaluation of cross-generator robustness in both detection and localization, establishing a medical-image-specific benchmark (Table 3).

## 3. Technical Approach
The work comprises two interlocking components: a controlled manipulation pipeline for 3D CT scans and a benchmarking protocol for forensics models.

A. Manipulation pipeline (Fig. 2; Section 2.1)
- Goal
  - Create realistic, local changes within a CT scan that emulate clinical scenarios: injecting large malignant nodules (diameter D > 10 mm) or removing/shrinking existing large nodules to appear benign/small (D < 8 mm).

- Why local and context-aware?
  - CTs are 3D volumes; local edits must blend with surrounding anatomical context to avoid obvious artifacts. The pipeline uses a 3D cube extracted from the scan where only the inner region is synthesized and the outer region conditions the generator to preserve contextual coherence (Fig. 2).

- Preprocessing steps (Fig. 2, top)
  - Candidate-site selection: Choose a location near existing nodules to reduce visual discontinuities (Section 2).
  - Cut a 3D physical cube of side 32 mm centered at the candidate site; rescale it to a standard `32×32×32` voxel grid. This normalization is needed because slice thickness and in-plane resolution vary across scanners (Section 2.1).
  - Equalize intensity values to a common range (standardization).
  - Mask the inner cube: zero out a `16 mm` inner region (center) so the generator must (re)synthesize only that part while using the outer ring as conditioning context.

- Generation models (three families; each takes the masked, equalized, and rescaled 3D cube as input; Fig. 2, grey "Generator")
  1) GAN-based inpainting with `CT-GAN` (3D Pix2Pix; [1], [10])
     - Pix2Pix is a conditional GAN that learns mappings from an input to a target image. Here, it’s adapted to 3D cubes.
     - Two separate models are trained: one specialized for injection (creating a large malignant nodule) and one for removal (replacing a large malignant nodule with a smaller benign-looking one) (Section 2.1).
  2) `3D CycleGAN` adaptation ([11])
     - CycleGAN learns to translate between two domains without paired data by enforcing cycle consistency.
     - Domains here:
       - Injection: translate from masked cubes to “cancerous” tissue.
       - Removal: translate from masked cubes to “non-cancerous” (benign/small-nodule) tissue.
     - The network is adapted to 3D cubes and used to generate the inner region conditioned on the surrounding context (Section 2.1).
  3) Diffusion Model (DM) inpainting with a 3D `DDPM` ([12–14])
     - Diffusion models incrementally denoise a sample from noise; here, a 3D U-Net denoiser is conditioned on the masked cube to ensure coherence with the context (Section 2.1).
     - Implementation details:
       - 3D U-Net denoiser (as in [14]) replaces the 2D denoiser.
       - Additional input channel supplies the masked cube as conditioning signal.
       - 2,000-step linear noise schedule (Section 2.1).

- Postprocessing steps (Fig. 2, bottom)
  - De-equalize intensities and rescale back to the scanner’s native physical voxel size.
  - Touch-up blending to improve seamless insertion into the original CT volume (Fig. 2).

- Design choices for realism (Section 2)
  - Rather than creating entirely new, isolated large nodules, the pipeline often:
    - Enlarges existing small nodules for injection.
    - Shrinks existing large nodules for removal.
  - New large nodules are placed near existing benign ones to reduce visual disruption (Fig. 1).

B. Dataset composition and splits
- Total manipulated samples: 8,577 (Table 1).
  - Injection: 6,238; Removal: 2,339.
  - By method (injection/removal):
    - Pix2Pix: 2,009 / 509
    - CycleGAN: 2,220 / 1,016
    - Diffusion Model: 2,009 / 814
- Pristine CTs: from LIDC-IDRI (1,018 scans, 1,010 patients) with annotated nodule positions and sizes (Section 2; [9]).
- Splits prevent leakage:
  - Patient-level split: 488 train, 100 validation, 150 test (Section 3).

C. Why these generators?
- The trio—conditional GAN (Pix2Pix/CT-GAN), unpaired translation GAN (CycleGAN), and diffusion (DDPM)—captures diverse synthesis mechanisms and forensic “fingerprints.” Evaluating across them stresses cross-generator generalization (Sections 1–3).

D. How the paper evaluates realism and downstream impact
- Check whether manipulations fool a medical nodule classifier (from [15]):
  - Apply only the classification module at the manipulated location.
  - Fig. 3 shows histograms:
    - In pristine scans, malignant nodules tend to get higher scores than benign ones (Fig. 3, top).
    - After manipulation, distributions “swap”: removed/shrunk malignant nodules look benign; injected/enlarged nodules look malignant (Fig. 3, bottom). This indicates successful deception of an automated diagnostic tool.

E. Benchmarking forensic detection and localization
- Detectors considered (Section 3; Table 3)
  - Localization-capable (produce pixel/voxel-wise maps): `U-Net` [17], `HP-FCN` [18], `ManTraNet` [19], `MVSS-Net` [20], `TruFor` [21].
  - Detection-only baseline: `Xception` [22] (also used inside some forensics models).
- Training protocol
  - Fine-tune models on M3Dsynth. Prior tests showed models trained on their original natural-image datasets perform poorly on CTs (Section 3).
  - Cross-generator evaluation: train on one generator type (e.g., Pix2Pix) and test on all three (Pix2Pix, CycleGAN, DM). This assesses generalization (Table 3).
- Metrics
  - Localization: `F1` and `IoU` (intersection-over-union) over 3D masks (Table 3, top).
  - Detection:
    - Balanced `Accuracy` with a fixed 0.5 threshold over the max slice-level score per volume (Table 3, bottom).
    - `Pd@1%`: set threshold to achieve 1% false alarms on pristine images, then measure detection probability. This simulates realistic operation where most scans are genuine (Table 3, bottom).

- Additional transferability probe: a synthetic-image detector from [16]
  - When trained on general-purpose images (ProGAN, StyleGAN2, LDM) and tested on M3Dsynth: ~50% accuracy—no signal (Table 2, top).
  - After fine-tuning on M3Dsynth: detection accuracy >90% across generators, while performance on general-purpose images drops (Table 2, bottom).

## 4. Key Insights and Innovations
- A large, medically grounded 3D tampering dataset
  - What’s new: 8,577 manipulated CT volumes with both injection and removal of lung nodules, spanning three different generator families (Table 1). Prior datasets were small (e.g., ~100 in [1]) and less diverse in generation methods.
  - Why it matters: Enables training and rigorous cross-generator benchmarking for 3D medical forensics—something existing general-purpose deepfake datasets cannot support (Sections 1–3).

- Context-conditioned 3D inpainting pipeline aligned with clinical realism
  - What’s different: The pipeline edits only the inner part of a 32 mm cube while conditioning on the outer ring (Fig. 2). It often transforms existing nodules (enlarge/shrink) and places injected nodules near existing benign ones to minimize visual discontinuities (Section 2).
  - Impact: Produces manipulations that preserve anatomical coherence, effectively fooling a trained medical classifier (Fig. 3).

- A 3D diffusion inpainting setup for medical volumes
  - What’s new: A 3D DDPM with a 3D U-Net denoiser conditioned on the masked cube, using a 2,000-step linear schedule (Section 2.1).
  - Significance: Adds a modern generative mechanism with different forensic signatures than GANs, broadening the dataset’s coverage and the challenge for detectors.

- Evidence that fine-tuning on M3Dsynth is necessary for success
  - Key result: A state-of-the-art synthetic-image detector trained on natural images collapses to chance on medical CTs, but exceeds 90% accuracy cross-generator after fine-tuning on M3Dsynth (Table 2).
  - Takeaway: Detectors need domain-specific data; medical imagery lacks camera/compression cues common in natural-image forensics (Section 3).

## 5. Experimental Analysis
- Evaluation methodology
  - Data: LIDC-IDRI pristine scans; 8,577 manipulated volumes split by patient (488/100/150 for train/val/test) to avoid leakage (Section 3; Table 1).
  - Tasks:
    - Does tampering fool medical classifiers? Use [15]’s classification component at edit locations; compare score distributions (Fig. 3).
    - Forensics:
      - Localization: produce 3D masks; compute F1 and IoU (Table 3, top).
      - Detection: balanced Accuracy with fixed threshold, plus Pd@1% with calibrated threshold (Table 3, bottom).
  - Cross-generator setup: Train on one generator; test on all three (Table 3).

- Main results
  1) Medical classifier deception (Fig. 3)
     - Quote: “After manipulation the removed/shrinked malignant nodules have lower scores and the injected/enlarged nodules have larger scores, with the two histograms exchanging roles.” In other words, manipulations flip the classifier’s benign/malignant assessment at the targeted site.

  2) General-purpose detector fails on medical CTs without fine-tuning (Table 2, top)
     - Example: Training on ProGAN/StyleGAN2/LDM yields strong results on those same general-purpose images (e.g., 99.9% and 100% on ProGAN/StyleGAN2), but accuracy drops to ~50% on Pix2Pix/CycleGAN/DM from M3Dsynth: 
       > “50.5 / 49.0 / 48.9” when testing on Pix2Pix/CycleGAN/DM after training on ProGAN (Table 2, top).

  3) Fine-tuning on M3Dsynth yields high, cross-generator detection accuracy (Table 2, bottom)
     - Example:
       > Training on Pix2Pix and testing on CycleGAN and DM: 96.6% and 95.8% accuracy; training on CycleGAN and testing on Pix2Pix and DM: 97.7% and 91.6% (Table 2, bottom).

  4) Localization performance is strong, especially for modern forensics models (Table 3, top)
     - `TruFor` [21] (transformer-based):
       > Train Pix2Pix → Test Pix2Pix/CycleGAN/DM: F1/IoU = 89.9/82.9, 68.1/55.5, 68.0/54.7.
       > Train CycleGAN → Test Pix2Pix/CycleGAN/DM: 79.0/70.1, 88.2/81.2, 65.0/54.1.
       > Train DM → Test Pix2Pix/CycleGAN/DM: 84.4/75.2, 76.9/66.7, 89.3/82.0.
     - `ManTraNet` [19] also performs well:
       > Train Pix2Pix → Test Pix2Pix/CycleGAN/DM: 87.0/79.1, 66.5/50.5, 61.4/45.5.
     - `U-Net` alone struggles comparatively (e.g., F1 often 35–57 across cross-generator cases).

  5) Detection performance depends on thresholding, but calibrated Pd@1% is strong for several models (Table 3, bottom)
     - Fixed-threshold accuracy can be misleadingly low for some models (e.g., `ManTraNet` shows ~52–57% accuracy), likely due to miscalibration when aggregating slice scores.
     - With realistic `Pd@1%` calibration (false alarm = 1% on pristine images), performance improves:
       - `TruFor`:
         > Train Pix2Pix → Test Pix2Pix/CycleGAN/DM: Pd@1% = 100 / 97.8 / 97.0.
         > Train CycleGAN → Test Pix2Pix/CycleGAN/DM: 95.9 / 99.4 / 89.1.
         > Train DM → Test Pix2Pix/CycleGAN/DM: 99.9 / 98.1 / 99.6.
       - `MVSS-Net` shows solid Pd@1% in aligned settings (e.g., 99.3 for CycleGAN→CycleGAN), though cross-generator Pd can drop.
       - `HP-FCN` lags behind consistently (e.g., Pd@1% often ~30–50%).

- Do experiments support the claims?
  - Yes, in several respects:
    - The medical-classifier deception is clearly shown by the histogram swap in Fig. 3.
    - Cross-domain failure of a general-purpose detector and recovery via fine-tuning on M3Dsynth (Table 2) strongly supports the need for domain-specific training.
    - Localization and calibrated detection metrics (Table 3) substantiate claims of strong performance and cross-generator generalization for the best models (TruFor, ManTraNet).
  - Caveats:
    - Detection accuracy with a fixed threshold is sometimes low; the authors argue threshold calibration matters and provide Pd@1% as a remedy (Table 3, bottom).
    - No human-reader study; only automated classifier deception is tested (Section 2.2).

- Ablations/robustness
  - The central robustness axis is cross-generator generalization (train on one generator, test on others). This is a meaningful stress test because real-world attackers could use different tools.
  - The paper does not report ablations on masking size, rescaling choices, or intensity equalization strategies.

## 6. Limitations and Trade-offs
- Scope limitations
  - Modality and anatomy: Only lung CT scans are considered; methods may not transfer directly to other organs or modalities (MRI, PET).
  - Manipulation types: Focused on nodules—mainly size changes (enlarge/shrink) and injection/removal—rather than broader pathologies or non-nodular edits (Section 2).

- Pipeline assumptions
  - Local-cube edit with fixed physical size (32 mm) and fixed network input size (32×32×32), plus equalization/rescaling (Section 2.1). Real-world manipulations might not follow these exact preprocessing steps.
  - Postprocessing “touch-up” is mentioned but not deeply detailed; different blending schemes could change forensic traces (Fig. 2).

- Data balance and generalization
  - Imbalance between injection (6,238) and removal (2,339) may influence detection biases (Table 1).
  - All pristine scans come from LIDC-IDRI; domain shifts across scanners/hospitals or protocols beyond this dataset are not evaluated.

- Computational considerations
  - Diffusion model uses 2,000 denoising steps (Section 2.1), which can be computationally heavy for training and inference on 3D volumes.

- Evaluation boundaries
  - No human-radiologist evaluation; deception is demonstrated only for an automated classifier (Fig. 3).
  - For one set of detector results (Table 2, top), the detection method assumes the forged area is known because it performs only detection, not localization—this underestimates real-world difficulty (Section 2.2).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a much-needed, large, medically realistic benchmark for 3D tampering in CT. This should shift medical image forensics from small, single-generator studies to robust cross-generator evaluation and training (Sections 1–3; Table 1, Table 3).
  - Shows that medical-image forensics cannot rely on detectors trained on natural images; domain-specific datasets and fine-tuning are essential (Table 2).

- Follow-up research enabled or suggested
  - Broaden the dataset:
    - Include additional modalities (MRI), body regions, and manipulation types beyond nodules (calcifications, fractures, devices).
    - Add more generative families (latent diffusion variants, masked autoencoders) and adversarially optimized attacks.
  - Robustness and generalization:
    - Study scanner/protocol domain shifts; include multiple acquisition settings and hospitals.
    - Explore self-supervised or domain-adaptive training to reduce reliance on labeled forged data.
    - Investigate calibration methods for detection thresholds and decision aggregation over 3D volumes.
  - Pipeline-level defenses:
    - Integrate detectors into PACS workflows; evaluate triage settings using Pd@low-FA rates (as in Pd@1%).
    - Consider provenance approaches (e.g., cryptographic signing of volumes) and cross-checks with radiology reports.

- Practical applications
  - Hospitals and vendors can use M3Dsynth to train detectors embedded in PACS or CT workstation software for real-time screening.
  - Regulatory and auditing bodies can adopt the benchmark to certify forensic tools for medical imaging.
  - Researchers developing medical AI can test robustness of diagnostic models against tampered inputs and design countermeasures.

Overall, M3Dsynth delivers a realistic, diverse 3D medical manipulation dataset and a rigorous benchmark showing that modern forensics models can detect and localize such tampering—provided they are trained on in-domain data. Figures 1–3 and Tables 1–3 supply the visual examples, pipeline details, dataset statistics, and quantitative evidence underpinning these conclusions.
