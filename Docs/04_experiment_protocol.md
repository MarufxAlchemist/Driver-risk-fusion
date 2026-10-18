# 04. Experiment Protocol

Purpose: decide how every number in the project is produced, **before** training starts, so results are comparable, honest and reproducible. Every novelty claim in `00_PROJECT_PLAN.md` maps to a row in section 5.

Status: DRAFT v0.1. Freeze target: end of week 2 (before P2 baselines).

---

## 1. Rules that apply to every experiment

1. **Split by driver**, never by window or trip (ADR-0005). No driver appears in more than one of train / val / test.
2. **Thresholds and hyperparameters are tuned on validation only.** The test split is touched once per final configuration.
3. **Normalization statistics** come from the training split only (`configs/norm_stats.json`).
4. **Three seeds minimum** (for example 0, 1, 2). Report mean and standard deviation. A difference smaller than one standard deviation is not claimed as an improvement.
5. **One config file per run** in `configs/`, copied into `experiments/<run_id>/config.yaml`. The run id includes the git commit hash.
6. **Same data, same splits, same metrics** for all models being compared.
7. **Log everything**: config, seed, commit, metrics (JSON), plots (PNG), model checkpoint path (not committed).
8. **Report failures** too. A negative ablation result goes in the report.

---

## 2. Datasets and splits

| Dataset | Use | Split scheme |
|---|---|---|
| UAH-DriveSet | Motion models, Stage 1 AE, aggressive / drowsy / normal | Leave-one-driver-out (few drivers) or grouped k-fold; report per-fold results |
| DMD | Vision models, drowsy / distracted | Subject-disjoint split, for example 70 / 15 / 15 by driver |
| Own recordings | **Paired fusion training and evaluation**, personalization | Driver-disjoint where possible; within-driver temporal split for personalization (see 2.2) |

### 2.1 Fusion evaluation on paired data

Because public sets are unpaired (`03_data_contract.md` 3.4):

| Setting | Training data | Test data | Claim it supports |
|---|---|---|---|
| F1 Paired (primary) | Own paired trips, driver-disjoint | Own paired trips of held-out drivers | Fusion beats single modality on truly paired data |
| F2 Pretrain + finetune | Public sets pretrain each branch, own data finetune the fusion | Held-out own drivers | Public data helps when own data is small |
| F3 Class-conditional pairing (bootstrapping) | UAH motion windows paired with DMD vision windows of the same class | UAH and DMD held-out drivers, **reported separately and labelled synthetic** | Only the robustness behaviour (graceful degradation), **never** headline accuracy |

F3 numbers must always carry the label "synthetic pairing" in tables and text.

### 2.2 Personalization split (within driver)

For each driver with at least `N_trips >= 4` trips (own data and UAH where possible):

- **Calibration set:** first `K` trips (default K = 2, normal driving only) used to build the personal baseline.
- **Evaluation set:** remaining trips, chronologically later.
- Compare the **same model** with (a) global threshold and (b) personalized baseline (shrinkage as in `02_architecture.md` 4.6), at equal detection recall.
- Report per-driver results as well as the mean, because personalization helps some drivers more than others.

If too few trips exist for some drivers, report them as "not enough data" instead of dropping them silently.

---

## 3. Metrics

### 3.1 Classification (Stage 2)

| Metric | Definition | Note |
|---|---|---|
| Macro-F1 | Mean of per-class F1 | Primary headline metric |
| Per-class recall and precision | | Recall matters most for safety classes |
| Confusion matrix | | Included for each final model |
| Balanced accuracy | | Secondary |
| ECE (expected calibration error) | 10 bins | Needed because alerts use probability thresholds |

### 3.2 Alerting (what a driver experiences)

| Metric | Definition |
|---|---|
| False alarms per hour (FA/h) | Alerts with no real abnormal event within `[t - 2 s, t + H]`, divided by hours of normal driving. Budget target: <= 6 FA/h at the chosen operating point (to be agreed with the guide) |
| Event recall | Fraction of annotated events that received an alert before offset |
| Lead time | `event_onset - alert_time` for events detected early; report median and IQR. Negative values count as late detections |
| Early-warning rate | Fraction of events with lead time >= 1 s |
| Alert latency | Time from window end to alert on the target device (ms) |

Operating point selection: choose `tau` on validation to hit the FA/h budget, then report recall and lead time on test at that `tau`. Also show the full ROC / PR and the recall-vs-FA/h curve.

### 3.3 Forecasting (Stage 3)

| Metric | Definition |
|---|---|
| PR-AUC per horizon H in {2, 3, 5} s | Event onset within H seconds |
| Lead time distribution | As above |
| Brier score | Calibration of risk probabilities |

Baselines for the forecaster: (a) "current-state" classifier probability reused as risk, (b) threshold on `|acc|` or PERCLOS, (c) persistence of the last window label.

### 3.4 Robustness (graceful degradation)

For a trained fusion model, evaluate with test inputs where one modality is masked (or degraded) for a fraction `r` of steps, `r` in {0, 0.1, 0.25, 0.5, 1.0}:

| Degradation | How it is simulated |
|---|---|
| Camera loss | `mask_v = False`, features zeroed |
| Camera noise / occlusion | Replace a fraction of frames with face-not-found, add landmark jitter |
| Low light proxy | Lower `face_conf` and drop detections randomly |
| IMU loss | `mask_m = False` |
| IMU noise | Gaussian noise at several SNR levels, random gaps |
| Phone moved | Random rotation of the IMU axes mid-trip |

Plot macro-F1 vs `r` for: sensor-only, camera-only, concatenation fusion, cross-attention fusion, cross-attention + modality dropout. The claim "stays reliable" is supported only if the last line degrades more slowly than concatenation and never falls below the better single modality by more than one standard deviation.

### 3.5 Efficiency

Parameters, FLOPs, model size (MB), inference time per 1 s step on: laptop CPU, laptop GPU (if any), and the target phone or edge device if available. Peak memory.

---

## 4. Models and baselines

| ID | Model | Input |
|---|---|---|
| B1 | Rule-based thresholds (jerk, PERCLOS, gaze) | Features |
| B2 | Sensor-only: LSTM / small transformer on `x_m` | Motion |
| B3 | Camera-only (feature path): LSTM on `x_v` | Vision features |
| B4 | Camera-only (3D-CNN) | Clips |
| B5 | Camera-only (2D CNN + LSTM) | Clips |
| B6 | Early-concat fusion (concat `x_v`, `x_m` per step, then LSTM) | Both |
| B7 | Late-concat fusion (separate encoders, concat embeddings) | Both |
| M1 | Cross-attention fusion, no modality dropout | Both |
| M2 | Cross-attention fusion + modality dropout (proposed) | Both |
| M3 | M2 + personalization | Both + driver history |
| M4 | M3 + forecaster | Both + history |

---

## 5. Ablation plan (claims -> evidence)

| ID | Claim | Experiment | Success criterion |
|---|---|---|---|
| A1 | Fusion beats single modalities | B2 vs B3 vs M2 on paired data (F1) | M2 macro-F1 > best of B2/B3 by more than 1 std |
| A2 | Cross-attention beats simple concatenation | B6, B7 vs M1 | M1 >= B6/B7 in macro-F1, and larger gap under degradation |
| A3 | Modality dropout provides graceful degradation | M1 vs M2 under 3.4 | M2 curve above M1 for r >= 0.25 |
| A4 | Personalization lowers false alarms | M2 global vs M3 at equal recall (2.2) | FA/h reduced, with per-driver table |
| A5 | Early prediction gives usable lead time | M4 vs forecaster baselines (3.3) | Median lead time >= 1 s at the FA/h budget |
| A6 | LSTM-AE adds value | M2 with vs without AE score as input | Report either way |
| A7 | 3D-CNN vs 2D+LSTM (optional) | B4 vs B5, then in fusion | Report accuracy, size, latency |
| A8 | Context features reduce false alarms | M3 with vs without speed/context | Report either way |
| A9 | Edge feasibility | 3.5 | End-to-end latency within budget on the CPU target |

Each ablation has an owner (see `07_phase_task_breakdown.md`) and writes its result into `experiments/ablations/A<n>.md` with a table, a plot and a two-line interpretation.

---

## 6. Training protocol (defaults, to be tuned on validation)

| Item | Default |
|---|---|
| Optimizer | AdamW, lr 1e-3 for feature models, 1e-4 for 3D-CNN fine-tuning, cosine decay |
| Batch | 64 windows (smaller for clip models) |
| Epochs | up to 100 with early stopping on validation macro-F1 (patience 10) |
| Loss | Cross-entropy, class weights or focal loss (gamma 2) if imbalance is high |
| Regularization | Dropout 0.2-0.3, weight decay 1e-2, modality dropout 0.15 in M2 |
| Augmentation | Motion: jitter, scaling, small rotation of axes. Vision features: landmark jitter, random face-not-found steps. Clips: flip is **not** allowed for head-yaw dependent labels without flipping the label feature too |
| Seeds | 0, 1, 2 (more if variance is high) |
| Hardware | Note GPU type and hours per run in the run log |

---

## 7. Reporting templates

Every final result table has:

`model | data setting (F1/F2/F3, own/UAH/DMD) | macro-F1 (mean ± std) | recall (aggressive / drowsy / distracted) | FA/h | median lead time | params | latency`

Every figure has axis labels with units, the seed count in the caption, and the dataset setting in the title.

---

## 8. Reproducibility checklist (per run)

- [ ] Config committed
- [ ] Commit hash recorded
- [ ] Seeds fixed (Python, NumPy, PyTorch, DataLoader workers)
- [ ] Splits come from `manifest.csv` with a stored split file hash
- [ ] Metrics JSON and plots saved under `experiments/<run_id>/`
- [ ] Result row added to `experiments/results.csv`

`make reproduce RUN=<run_id>` (created in P5) re-runs a configuration and compares metrics within tolerance.

---

## 9. Threats to validity (must be discussed in the final report)

| Threat | Mitigation |
|---|---|
| Very few drivers in UAH and own data | Report per-driver variance, grouped CV, do not over-claim significance |
| Simulated or posed drowsiness and distraction | State clearly; use DMD annotations for real behaviours; avoid claims about real fatigue |
| Cross-dataset domain shift | Report F3 results separately and labelled synthetic |
| Label noise in trip-level labels | Event-level evaluation for alerts, trip-level only for classification, state both |
| Threshold tuning leakage | Thresholds on validation only |
| Single phone model and vehicle | Describe setup; add a second phone or car if time permits |
