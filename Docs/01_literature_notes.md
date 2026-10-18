# 01. Literature Notes

Purpose: one place that says what each reference is, why it is in the project, and what we can and cannot claim from it. Used for the synopsis, the reports, and guide Q&A.

**Honesty rule:** the "what we claim" column only contains claims our notes support. Before the synopsis, each paper's owner reads the paper in full and fills the "Verified by reading" checkbox. Do not cite a number from a paper you have not opened.

---

## 1. Canonical reference list (R1-R11)

IEEE format. These IDs (R1-R11) are the canonical numbering for all repo docs.

| ID | Citation | Link |
|---|---|---|
| R1 | A. Kashevnik, R. Shchedrin, C. Kaiser, A. Stocker, "Driver distraction detection methods: A literature review and framework," *IEEE Access*, vol. 9, pp. 60063-60076, 2021. | https://doi.org/10.1109/ACCESS.2021.3073599 |
| R2 | J. Wang et al., "A survey on driver behavior analysis from in-vehicle cameras," *IEEE Trans. Intell. Transp. Syst.*, vol. 23, no. 8, pp. 10186-10209, 2022. | https://doi.org/10.1109/TITS.2021.3126231 |
| R3 | Z. Wang, Y. Wu, Q. Niu, "Multi-sensor fusion in automated driving: A survey," *IEEE Access*, vol. 8, pp. 2847-2868, 2020. | https://doi.org/10.1109/ACCESS.2019.2962554 |
| R4 | I. D. Bin Rozaimi et al., "Driver drowsiness detection using deep learning models based on different camera positions," *ICoICT*, 2023, pp. 611-616. | https://doi.org/10.1109/ICoICT58202.2023.10262759 |
| R5 | N. Kose, O. Kopuklu, A. Unnervik, G. Rigoll, "Real-time driver state monitoring using a CNN based spatio-temporal approach," *ITSC*, 2019, pp. 3236-3242. | https://doi.org/10.1109/ITSC.2019.8917460 |
| R6 | D. Tan et al., "Driver distraction behavior recognition for autonomous driving: Approaches, datasets and challenges," *IEEE Trans. Intell. Veh.*, vol. 9, pp. 8000-8026, 2024. | https://ieeexplore.ieee.org/document/10540299 |
| R7 | J. Kuang et al., "Spatio-temporal pruning: A generic method for edge-based 3D distracted driver recognition," *IEEE Trans. Mobile Comput.*, early access, 2026. | https://ieeexplore.ieee.org/document/11498637 |
| R8 | X. Liao et al., "A review of personalization in driving behavior: Dataset, modeling, and validation," *IEEE Trans. Intell. Veh.*, vol. 10, pp. 1241-1262, 2025. | https://ieeexplore.ieee.org/document/10591381 |
| R9 | X. Chen, Y. Wang, X. Sun, Y. Cai, "Unsupervised cross-scenario abnormal driving behavior recognition using smartphone sensor data," *IEEE Internet Things J.*, early access, 2023. | https://ieeexplore.ieee.org/document/10365683 |
| R10 | J. D. Ortega et al., "DMD: A large-scale multi-modal driver monitoring dataset for attention and alertness analysis," *ECCV 2020 Workshops*. | https://arxiv.org/abs/2008.12085 , dataset: https://dmd.vicomtech.org/ |
| R11 | E. Romera, L. M. Bergasa, R. Arroyo, "Need data for driver behaviour analysis? Presenting the public UAH-DriveSet," *ITSC*, 2016. | https://doi.org/10.1109/ITSC.2016.7795584 , reader: https://github.com/Eromera/uah_driveset_reader |

### Numbering mismatch to fix in the first-presentation deck

The deck's reference slide uses a different order and lists only 9 items, while other slides cite 10 and 11.

| Canonical | Paper | Number on deck references slide |
|---|---|---|
| R1 | Kashevnik | [1] |
| R2 | Wang (survey, cameras) | [2] |
| R3 | Wang (sensor fusion) | [3] |
| R4 | Rozaimi | [4] |
| R5 | Kose | [5] |
| R6 | Tan | [9] |
| R7 | Kuang | [6] |
| R8 | Liao | [7] |
| R9 | Chen | [8] |
| R10 | DMD | missing |
| R11 | UAH-DriveSet | missing |

Problems visible in the deck:
- The Methodology slide cites UAH-DriveSet as [11] and DMD as [10], but the references slide ends at [9].
- The Problem Statement slide cites [8] for "fixed thresholds treat every driver alike". On the deck's own list [8] is the Chen smartphone paper; the personalization review is [7] (Liao).

Fix before the next presentation: add DMD and UAH-DriveSet to the references slide and renumber so that the citations match.

---

## 2. Story arc (how the references defend the project)

```
Vision-only works but has blind spots      R1 R2 R4 R5
        |
Fusion works; phones are viable sensors     R3 R9
        |
3D-CNN beats 2D-CNN and can run on edge     R6 R7
        |
Personalization is a named open direction   R8
        |
Datasets we can actually train on           R10 R11
        |
GAP: nobody combines graceful degradation + per-driver baseline
     + early prediction in one edge-deployable unit
```

---

## 3. Per-paper notes

Template per paper: what it is, why we need it, what we claim, limits, likely guide question.

### R1 Kashevnik et al. (2021): distraction detection review
- **What:** literature review and framework of driver distraction detection methods.
- **Why we need it:** establishes the taxonomy of distraction (visual, manual, cognitive) and the limits of camera-only approaches.
- **We claim:** cognitive distraction is hard to see from a camera alone, which motivates adding motion signals.
- **Limits:** a review, so no experiments of its own that we can reuse.
- **Guide question:** "Why not just use a camera?" Answer: the camera misses cognitive distraction and degrades with lighting and occlusion; vehicle motion provides independent evidence.
- Verified by reading: [ ]

### R2 Wang et al. (2022): in-vehicle camera behaviour survey
- **What:** survey of driver behavior analysis from in-vehicle cameras.
- **Why:** vision-only tradition, its tasks, datasets and failure modes (lighting, occlusion, camera position).
- **We claim:** vision-only systems share known failure modes that a second modality can cover.
- Verified by reading: [ ]

### R3 Wang, Wu, Niu (2020): multi-sensor fusion survey
- **What:** survey of multi-sensor fusion in automated driving.
- **Why:** background for fusion levels (early, feature-level, late) and why synchronization is hard.
- **We claim:** fusing complementary sensors gives a more complete picture than any single sensor.
- **Limits:** about automated driving perception (lidar, radar, camera), not driver monitoring. Cite it for the fusion principle only.
- **Guide question:** "Why cross-attention rather than concatenation?" Answer: concatenation weights all sources equally and breaks when one is missing; cross-attention lets each stream query the other and can ignore a degraded one (we test this in ablation A3 vs A4/A5).
- Verified by reading: [ ]

### R4 Rozaimi et al. (2023): drowsiness and camera position
- **What:** deep-learning drowsiness detection compared across camera positions.
- **Why:** shows camera placement changes accuracy, which matters for a phone on a dashboard or windshield.
- **We claim:** placement variability is a real deployment issue; our data collection protocol fixes and records mount position.
- Verified by reading: [ ]

### R5 Kose et al. (2019): CNN spatio-temporal driver state
- **What:** real-time driver state monitoring using a CNN-based spatio-temporal approach.
- **Why:** early evidence that modeling time beats frame-by-frame classification, and that this can run in real time.
- Verified by reading: [ ]

### R6 Tan et al. (2024): distraction recognition review
- **What:** approaches, datasets, challenges for distraction recognition.
- **Why:** supports "3D CNN outperforms 2D CNN" for behaviour recognition and lists the datasets.
- **We claim:** 3D CNNs are a justified backbone candidate; we still validate against a 2D+LSTM baseline (ADR-0004).
- Verified by reading: [ ]

### R7 Kuang et al. (2026): spatio-temporal pruning for edge 3D CNN
- **What:** a pruning method for edge-based 3D distracted driver recognition.
- **Why:** answers the objection "3D CNNs are too heavy for a phone."
- **We claim:** pruned 3D CNNs can run on edge hardware. Early access paper, so check final publication details before the final report.
- Verified by reading: [ ]

### R8 Liao et al. (2025): personalization in driving behavior
- **What:** review of dataset, modeling and validation for personalized driving behavior.
- **Why:** our strongest novelty anchor. It frames personalization as an open direction.
- **We claim:** per-driver modelling is recognised as an open problem; most deployed systems use one fixed global threshold.
- **Limits:** do not claim "no one has done personalization". Claim: we combine it with fusion and early prediction on edge.
- **Guide question:** "Is personalization really new?" Answer: it is an open direction per R8; our contribution is combining it with fused sensing and early prediction, with measured benefit.
- Verified by reading: [ ]

### R9 Chen et al. (2023): unsupervised cross-scenario abnormal driving, smartphone sensors
- **What:** closest prior work. Smartphone sensor data, unsupervised abnormal driving recognition across scenarios.
- **Why:** proves smartphone sensors are viable and unsupervised methods work.
- **Difference to us:** sensors only (no camera), no per-driver baseline, no early prediction, no graceful degradation. Read this paper first and write the exact difference table below.
- Verified by reading: [ ]

Difference table (fill after reading R9):

| Aspect | R9 | This project |
|---|---|---|
| Modalities | | |
| Learning | | |
| Personalization | | |
| Prediction horizon | | |
| Handles sensor failure | | |
| Edge deployment | | |

### R10 DMD (Ortega et al., 2020): driver monitoring dataset
- **What:** large-scale multi-modal driver monitoring dataset for attention and alertness analysis.
- **Why:** our vision-side training data (distraction, drowsiness related behaviours).
- **Check on download:** exact label set, camera views available, modalities (RGB / depth / IR), license terms, per-driver split. Record results in `03_data_contract.md` section 3.
- Verified by reading: [ ]

### R11 UAH-DriveSet (Romera et al., 2016)
- **What:** public naturalistic driving dataset recorded with a smartphone app.
- **Why:** our motion-side training data (accelerometer, GPS, derived events).
- **Facts to verify on download (do not assume):** number of drivers, behaviours (normal, drowsy, aggressive), road types, sampling rates of raw accelerometer and GPS, what the video shows (we expect the road ahead rather than the driver's face), file formats. Record results in `03_data_contract.md` section 3.
- **Known caveat to confirm:** drowsy and aggressive behaviours are driver-performed styles per trip, so labels are trip-level, not event-level.
- Verified by reading: [ ]

---

## 4. Likely guide questions

| Question | Short answer | Backed by |
|---|---|---|
| What is new here? | Graceful-degradation fusion + per-driver baseline + early prediction with explanations, in one edge-deployable system | R3, R8, R9, ablation plan |
| Why not camera only? | Misses vehicle-motion evidence and fails under lighting, occlusion, placement | R1, R2, R4 |
| Why not sensors only? | Misses eye closure, gaze, head pose | R1, R9 |
| Why 3D CNN? | Models time; reported to beat 2D CNN; can be pruned for edge. We validate it against 2D+LSTM | R5, R6, R7 |
| Can it run on a phone? | Edge-first design, ONNX export, latency measured in P5 | R7 |
| Where does the data come from? | UAH-DriveSet, DMD, plus our own paired recordings | R10, R11 |
| How do you evaluate personalization? | Within-driver temporal split: calibrate on first trips, test on later ones | `04_experiment_protocol.md` |
| What if the camera fails? | Modality dropout training; we plot performance vs modality loss | `04_experiment_protocol.md` |

---

## 5. Reading plan

Habit target: each member reads one assigned paper per week and writes a 5-line note (problem, method, data, result, limit) into this file under their paper.

| Order | Paper | Reader |
|---|---|---|
| 1 | R9 (closest prior work) | Maruf |
| 2 | R8 | Maruf |
| 3 | R10, R5 | Anuja |
| 4 | R11 | Tanishq |
| 5 | R3, R6 | Tanishq |
| 6 | R7, R4 | Anuja |
| 7 | R1, R2 | Gautam |
