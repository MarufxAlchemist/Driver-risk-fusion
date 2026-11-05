# 00. Project Plan

**Title:** Personalized Driver Behavior Monitoring and Early Risk Prediction using 3D-CNN and Cross-Attention Multi-Sensor Fusion
**Team:** Anuja Naik, Tanishq Basagare, Maruf Nadaf, Gautam Iyengar
**Guide:** Mrs. Urmila V. Patil
**Status:** DRAFT v0.2 (planning only, no code yet)

Changes from v0.1:
- Corrected the dataset description: UAH-DriveSet labels are *normal / drowsy / aggressive* driving styles; it has **no distraction class**. Distraction comes from DMD.
- Added the **paired-data problem** (no public dataset has driver-face video and phone IMU for the same trip). This raises the priority of our own recordings. See ADR-0003 and `09_risk_register.md` R1.
- Added the full document set (01-10, ADRs, report outlines).

---

## 1. Guiding principles

1. **Documentation before code.** Every module gets a short spec (inputs, outputs, owner) before implementation.
2. **Interfaces first, models second.** Four people can only work in parallel if the data formats between modules are frozen early (`03_data_contract.md`).
3. **Baselines before novelty.** Camera-only and sensor-only baselines work before fusion, personalization and forecasting are built on top.
4. **Always have a working demo.** Every phase ends with something runnable end to end, even if crude.
5. **Mock everything upstream.** Each person ships a stub (fake output in the agreed format) by the end of week 2 so nobody is blocked.
6. **Claims need numbers.** Every novelty claim maps to one ablation or metric in `04_experiment_protocol.md`.

---

## 2. Repository skeleton

```
driver-risk-fusion/
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml               # single env definition
├── Makefile                     # make setup / lint / test / train / serve / demo-stub
├── .gitignore                   # data/, checkpoints/, *.pt, *.onnx, .env
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/ci.yml         # lint + unit tests on PR
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/
│   ├── 00_PROJECT_PLAN.md       # this file
│   ├── 01_literature_notes.md
│   ├── 02_architecture.md
│   ├── 03_data_contract.md      # FREEZE EARLY
│   ├── 04_experiment_protocol.md
│   ├── 05_api_contract.md
│   ├── 06_data_collection_protocol.md
│   ├── 07_phase_task_breakdown.md
│   ├── 08_team_workflow.md
│   ├── 09_risk_register.md
│   ├── 10_glossary.md
│   ├── decisions/               # ADR-0001 ... 
│   └── reports/                 # synopsis, mid-term, final drafts
├── configs/                     # YAML: data, model, train, personalization, serve
├── data/                        # git-ignored; README explains how to fetch
│   ├── raw/{uah_driveset,dmd,own}/
│   ├── interim/
│   └── processed/{uah,dmd,own}/ # one parquet per trip + manifest.csv
├── src/driverisk/
│   ├── data/                    # loaders, timestamp sync, windowing, datasets
│   ├── features/
│   │   ├── vision.py            # MediaPipe: EAR, MAR, PERCLOS, head pose, gaze
│   │   └── motion.py            # IMU/GPS: jerk, yaw rate, speed variance, lateral drift
│   ├── models/
│   │   ├── lstm_autoencoder.py  # Stage 1: learns "normal"
│   │   ├── video_backbone.py    # 3D-CNN and 2D+LSTM (ablation pair)
│   │   ├── fusion.py            # Stage 2: cross-attention + modality dropout
│   │   ├── forecaster.py        # Stage 3: risk H seconds ahead
│   │   └── baselines.py         # camera-only, sensor-only, concat fusion
│   ├── personalization/         # per-driver baseline, calibration, shrinkage
│   ├── explain/                 # reason generation for alerts
│   ├── training/                # trainers, losses, callbacks
│   ├── evaluation/              # metrics, ablations, lead-time, false-alarm rate
│   ├── inference/               # real-time pipeline, ONNX export
│   ├── serving/                 # FastAPI app, schemas
│   └── utils/
├── dashboard/                   # web UI, trip analytics
├── notebooks/                   # EDA only
├── scripts/                     # download_data.sh, preprocess.py, train.py, demo.py
├── tests/                       # unit tests, mock fixtures
└── experiments/                 # run logs and result tables (small files only)
```

Branching: `main` is always demo-able. Work on `feat/<area>-<name>`, PR into `dev`, one reviewer, `dev` merges to `main` at each phase end. Details in `CONTRIBUTING.md`.

---

## 3. Phases

Assumes about 24 weeks starting late October 2026. Adjust once the college mid-term and final dates are confirmed.

| Phase | Weeks | Goal | Exit criterion |
|---|---|---|---|
| **P0 Foundations** | 1-2 | Repo, env, docs, data access, frozen contract, stubs, dataset verification | Everyone can clone, run `make setup`, run the stub pipeline end to end |
| **P1 Data and features** | 3-6 | Clean, synced, windowed data; feature extractors; **start own data collection** | Processed UAH + DMD windows load via one DataLoader; extractors unit-tested; first own trips recorded |
| **P2 Baselines** | 5-8 | Camera-only, sensor-only, LSTM-AE (Stage 1) | Baseline numbers logged per config, reproducible |
| **P3 Fusion** | 8-12 | Cross-attention fusion + modality dropout | Fused beats both single-modality baselines on paired data; degrades gracefully when one modality is masked |
| **P4 Personalization and prediction** | 11-16 | Per-driver baseline, risk forecaster, explanations | Personalized beats global threshold on false alarms/hour; measurable lead time |
| **P5 System integration** | 14-19 | Real-time pipeline, ONNX export, FastAPI + dashboard | Live demo from a recorded trip and from webcam + phone IMU |
| **P6 Evaluation and own data** | 17-22 | Complete own dataset, all ablations, final numbers | All tables and plots regenerate from one script |
| **P7 Reports and polish** | 21-24 | Synopsis, mid-term, final report, slides, demo video | Report submitted, repo cleaned, README complete |

Phases overlap on purpose. Backend and dashboard (P5 work) start early against mock model outputs. Task-level detail: `07_phase_task_breakdown.md`.

---

## 4. Priorities (MoSCoW)

**Must have**
- UAH-DriveSet + DMD preprocessing and windowing
- Vision features (EAR, MAR, PERCLOS, head pose) and motion features
- Camera-only and sensor-only baselines
- **A small paired own dataset** (driver-face video + phone IMU/GPS, same trip) to train and test fusion honestly
- Cross-attention fusion classifier with ablation vs concatenation
- Evaluation protocol: macro-F1, per-class recall, false alarms per hour
- Working demo (offline replay acceptable)

**Should have (the novelty claims)**
- LSTM autoencoder Stage 1
- Modality dropout / graceful degradation experiment
- Per-driver personalization with measured benefit
- Risk forecaster with lead-time metric
- FastAPI dashboard with explainable alerts

**Could have**
- 3D-CNN vs 2D+LSTM comparison (3D-CNN is a candidate, not a commitment; ADR-0004)
- ONNX export with latency benchmark on a phone or laptop CPU
- Larger own dataset (more drivers)

**Won't (Future Scope)**
- CARLA simulation, V2V, multi-camera, ADAS integration, fleet cloud

Scope guard: if P3 slips by more than 2 weeks, cut 3D-CNN to a Could and use the MediaPipe-feature path only. Decide at the end of week 10.

---

## 5. Team split (proposal, to be confirmed)

Each person owns one workstream and has a named backup. Roles can rotate at phase boundaries.

| Workstream | Owner | Backup | Owns |
|---|---|---|---|
| **A. Vision pipeline** | Anuja | Gautam | DMD loader, MediaPipe features, video backbone (3D-CNN / 2D+LSTM), camera-only baseline |
| **B. Motion pipeline** | Tanishq | Maruf | UAH loader, IMU/GPS cleaning, motion features, LSTM-AE (Stage 1), sensor-only baseline |
| **C. Fusion, personalization, forecasting** | Maruf | Anuja | Cross-attention fusion, modality dropout, personalization, forecaster, explanations, experiment protocol |
| **D. Platform and delivery** | Gautam | Tanishq | Repo, CI, configs, FastAPI, dashboard, ONNX export, data-collection logistics, report assembly |

Why this runs in parallel: A and B emit features in the format fixed by `03_data_contract.md`; C builds against stub features from week 2; D builds the API and dashboard against mock model outputs. Full detail: `08_team_workflow.md`.

---

## 6. Document set and reading order

| # | Doc | Read when |
|---|---|---|
| 1 | `00_PROJECT_PLAN.md` | Everyone, first |
| 2 | `10_glossary.md` | Anyone new to the domain |
| 3 | `01_literature_notes.md` | Before writing report sections or answering guide questions |
| 4 | `02_architecture.md` | Before designing any module |
| 5 | `03_data_contract.md` | Before writing any data or feature code |
| 6 | `04_experiment_protocol.md` | Before training anything |
| 7 | `05_api_contract.md` | Workstreams C and D |
| 8 | `06_data_collection_protocol.md` | Before recording anything |
| 9 | `07_phase_task_breakdown.md`, `08_team_workflow.md` | Weekly planning |
| 10 | `09_risk_register.md` | Weekly sync, review every two weeks |
| 11 | `decisions/` | When a "why did we do it this way" question comes up |

---

## 7. Open decisions (team call needed)

| ID | Decision | Default we proceed with | Deadline |
|---|---|---|---|
| D1 | Window length and stride | 5 s window, 1 s stride (multi-scale PERCLOS at 60 s as a feature) | End of week 1 |
| D2 | Unified label map | See `03_data_contract.md` section 5 | End of week 1 |
| D3 | Split strategy | Split by driver (ADR-0005) | End of week 1 |
| D4 | Compute access (college GPU lab, Colab, Kaggle) | Colab/Kaggle until confirmed | End of week 1 |
| D5 | Pinned versions (Python, PyTorch, MediaPipe) | Python 3.11, latest stable PyTorch at repo creation, pinned in `pyproject.toml` | Week 1 |
| D6 | Phone sensor-logging app for own data | Choose one that exports timestamped CSV (verify on both iOS and Android) | Week 2 |
| D7 | Fusion training data strategy | ADR-0003 (own paired data + bootstrapping from public sets) | Week 2 |
| D8 | License | Decide with guide; default MIT for code | Week 3 |
