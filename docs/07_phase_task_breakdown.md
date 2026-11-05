# 07. Phase Task Breakdown

Each task has an ID `P<phase>-<workstream><n>`, an owner, a deliverable and a "done when" check. Create one GitHub issue per task and put the ID in the title. Workstreams: **A** Anuja (vision), **B** Tanishq (motion), **C** Maruf (fusion / personalization / forecasting), **D** Gautam (platform / delivery).

Effort scale: S = up to 1 day, M = 2-4 days, L = 1-2 weeks.

Dependencies are written as `after <ID>`. Anything without a dependency can start immediately.

---

## P0: Foundations (weeks 1-2)

Goal: everyone can clone, install, run the stub pipeline, and the contracts are frozen.

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P0-D1 | Create GitHub repo, branch protection (`main`, `dev`), issue labels, project board | D | S | Repo live | All four have push access; PR needed to merge to `dev` |
| P0-D2 | Repo scaffold: folders, `pyproject.toml`, Makefile, pre-commit, `.gitignore` | D | M | `make setup` works on Mac, Windows laptop | Fresh clone installs in under 10 minutes on two OSes |
| P0-D3 | CI: lint (ruff), tests (pytest) on PR | D | S | `.github/workflows/ci.yml` | Green check on a sample PR |
| P0-D4 | Commit docs 00-10 + ADRs into `docs/` | D | S | Docs on `main` | README links resolve |
| P0-C1 | Run team call on D1-D8 (`00_PROJECT_PLAN.md` section 7), record decisions as ADRs | C | S | ADR updates | All "Open decisions" have a value |
| P0-C2 | Freeze data contract v1.0 (after P0-C1 and P0-A1/B1) | C | M | `03_data_contract.md` v1.0 | All four sign off in the PR |
| P0-A1 | Register for and download DMD, inspect views, labels, license; fill `03_data_contract.md` 3.2 | A | M | Verified table + sample notebook | Section 3.2 has no unchecked "Verified" boxes |
| P0-B1 | Download UAH-DriveSet, run the reader, fill `03_data_contract.md` 3.1 (confirm drivers, rates, video view, labels) | B | M | Verified table + sample notebook | Section 3.1 fully verified |
| P0-D5 | Choose phone logging app (D6): test export format, sync accuracy, battery on both phone types | D | M | ADR + test recording | A 5-minute test trip with usable IMU and video |
| P0-D6 | Guide meeting: approve data collection protocol and consent form | D | S | Signed-off protocol | Approval recorded in meeting notes |
| P0-A2 | Stub: `make_stub_vision.py` | A | S | `tests/fixtures/` | Passes the contract validator |
| P0-B2 | Stub: `make_stub_motion.py` | B | S | `tests/fixtures/` | Passes the contract validator |
| P0-C3 | Contract validator + `WindowDataset` skeleton + unit tests on stubs | C | M | `data/datasets.py`, `data/validate.py` | CI green; batch shapes match section 7 of the contract |
| P0-C4 | Toy fusion forward pass on stub batches | C | S | Notebook or test | Loss decreases on a toy rule |
| P0-D7 | Mock FastAPI server with scripted alerts | D | M | `scripts/mock_server.py` | Dashboard shell can display the scripted sequence |
| P0-ALL | Read R9 and R8; write 5-line notes into `01_literature_notes.md` | C | S | Notes | Boxes ticked |

**Phase gate:** `make demo-stub` runs stub data -> stub features -> toy fusion -> mock API -> dashboard shell.

---

## P1: Data and features (weeks 3-6)

Goal: real data flows through the contract; own data collection starts.

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P1-B1 | UAH loader -> per-trip parquet (10 Hz, contract columns) | B | L | `data/uah.py`, processed UAH data | Validator passes on all trips |
| P1-B2 | Motion features: gravity removal, axis alignment, jerk, yaw rate, speed variance | B | L | `features/motion.py` + tests | Unit tests on synthetic signals; output ranges sane |
| P1-B3 | Event detector for harsh accel / brake / turn / lane weave with thresholds in `configs/events.yaml` | B | M | `features/events.py` | Event counts per UAH behaviour sensible (aggressive trips have more) |
| P1-A1 | MediaPipe vision features: EAR, MAR, head pose, gaze proxies, face confidence | A | L | `features/vision.py` + tests | Features on 10 sample clips plotted and sanity-checked |
| P1-A2 | Causal PERCLOS (30 s, 60 s), blink and eye-closed detection with per-driver EAR threshold | A | M | in `vision.py` | Matches manual count of closures on 5 annotated clips |
| P1-A3 | DMD loader -> per-trip parquet, label map to classes | A | L | `data/dmd.py`, processed DMD data | Validator passes; class counts report |
| P1-C1 | Label map config files `configs/label_map_*.yaml`, reviewed with team | C | S | YAML files | Merged after team review |
| P1-C2 | Split script: driver-disjoint splits + manifest, frozen split file | C | M | `scripts/make_splits.py`, `splits.json` | No driver overlap test passes |
| P1-C3 | Normalization stats + dataset augmentations | C | S | `norm_stats.json`, augmentation module | Applied in `WindowDataset` |
| P1-D1 | Own-data pipeline script: sync via tap events, resample, write parquet | D | L | `scripts/process_own.py` | Pilot trip reaches contract format with offset < 100 ms |
| P1-D2 | Pilot recordings: 2 drivers x 2 trips (after P0-D6) | D (all record) | M | Raw data + trip sheets | Quality checks (protocol section 9) pass |
| P1-D3 | Dashboard shell: trip list, live risk gauge, alert feed using the mock server | D | L | `dashboard/` | Displays the mock sequence |
| P1-D4 | Set up experiment tracking (CSV + MLflow or W&B) and the `experiments/` convention | D | S | Docs + helper | A dummy run is logged |

**Phase gate:** a single `DataLoader` yields batches from UAH, DMD, and a pilot own trip with correct shapes and masks. Data report (counts per class per driver) committed to `experiments/data_report.md`.

---

## P2: Baselines (weeks 5-8)

Goal: honest baseline numbers before any novelty.

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P2-B1 | B1 rules baseline + B2 sensor-only LSTM on UAH | B | M | Run logs | Macro-F1 table with 3 seeds |
| P2-B2 | Stage 1 LSTM-AE on normal windows, reconstruction error analysis | B | L | `models/lstm_autoencoder.py` | Error separates normal from aggressive on UAH validation (report AUROC) |
| P2-A1 | B3 camera-only (feature LSTM) on DMD | A | M | Run logs | Macro-F1 table with 3 seeds |
| P2-A2 | B5: 2D CNN + LSTM on clips (small subset first) | A | L | `models/video_backbone.py` | Trains and logs on a DMD subset |
| P2-A3 | B4: 3D-CNN on clips, check GPU time per epoch, decide go / no-go by week 8 (ADR-0004) | A | L | Run logs + ADR update | Decision recorded |
| P2-C1 | Metrics library: macro-F1, FA/h, lead time, ECE, robustness curves | C | L | `evaluation/` + tests | Tested on synthetic cases |
| P2-C2 | B6 / B7 concat fusion baselines on paired setting (pilot data) and on F3 synthetic pairing | C | M | Run logs | Runs end to end; results labelled by setting |
| P2-C3 | Training loop, config system, seeding, checkpointing | C | M | `training/` | Reproducible runs |
| P2-D1 | Continue recording (2 sessions per week) | D (all) | L | More own trips | Reach 8 trips by week 8 |
| P2-D2 | Annotation tool/process and first labels for own trips | D (+A) | M | Event JSONs | 20% overlap kappa reported |
| P2-D3 | Baseline results page in the dashboard or `experiments/results.md` | D | S | Results summary | Updated by script |

**Phase gate:** `experiments/results.csv` contains B1-B7 numbers with mean and std, and an explicit go/no-go note on the 3D-CNN.

---

## P3: Fusion (weeks 8-12)

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P3-C1 | Cross-attention fusion `models/fusion.py` with masks, missing tokens, gates | C | L | Module + tests | Masked-stream tests pass (output unchanged by masked values) |
| P3-C2 | Modality dropout training (M2) | C | M | Config + runs | Runs with p in {0, 0.1, 0.15, 0.3} |
| P3-C3 | Ablations A1, A2 | C | L | `experiments/ablations/A1.md`, `A2.md` | Tables, plots, interpretation |
| P3-C4 | Robustness evaluation (A3): degradation sweeps | C | L | `A3.md` + plots | Curves for 5 models |
| P3-A1 | Vision embedding export for fusion (V-feat, plus V-3D if go) | A | M | Embeddings or modules | Plug into fusion with the contract shapes |
| P3-A2 | Vision-side degradation simulators (occlusion, jitter, low light proxy) | A | M | `evaluation/degrade_vision.py` | Used by A3 |
| P3-B1 | Motion-side degradation simulators (noise, gaps, axis rotation) | B | M | `evaluation/degrade_motion.py` | Used by A3 |
| P3-B2 | Add AE score as optional fusion input; ablation A6 | B | M | `A6.md` | Table either way |
| P3-D1 | Real-time inference engine skeleton (stream in, windows, model, alert out) with a replay mode | D | L | `inference/` | Replays a processed trip in real time |
| P3-D2 | WebSocket integration with the dashboard, real alert objects (stub model) | D | M | Live dashboard with mock model | Alert JSON matches the API contract |
| P3-D3 | Recording sessions continue; dataset v0.5 card | D | M | Data card | Updated |

**Phase gate:** fused model results with three seeds; decision on whether A1-A3 claims hold; if they do not, discuss with the guide and re-scope (see risk register R1 and R12).

**Mid-term report draft starts here** (`docs/reports/README.md`).

---

## P4: Personalization and prediction (weeks 11-16)

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P4-C1 | Personal baseline module with shrinkage and drift control | C | L | `personalization/` + tests | Unit tests on synthetic drivers |
| P4-C2 | Personalization evaluation protocol run (A4), per-driver tables | C | L | `A4.md` | FA/h at equal recall reported |
| P4-C3 | Forecaster: label generation (`y_risk`), model, training | C | L | `models/forecaster.py` | PR-AUC per horizon logged |
| P4-C4 | Alerting rule: debounce, thresholds from FA/h budget, lead-time metric (A5) | C | M | `A5.md` | Median lead time and FA/h reported |
| P4-C5 | Explanation module and templates | C | M | `explain/` + `templates.yaml` | Reasons produced for all alert types |
| P4-A1 | Per-driver EAR calibration routine for the app | A | M | In `vision.py` | Reduces eye-closed errors vs fixed threshold |
| P4-B1 | Context features (speed bucket) and ablation A8 | B | M | `A8.md` | Table either way |
| P4-B2 | Per-driver motion normalization (acceleration style) | B | M | In `motion.py` | Used by baseline module |
| P4-D1 | Dashboard: driver baseline view, "why" panel, personalized vs global comparison | D | L | `dashboard/` | Shows evidence from alert JSON |
| P4-D2 | Calibration flow in the edge app (first trips) | D | M | `inference/calibration.py` | Driver status moves calibrating -> ready |

**Phase gate:** A4 and A5 results exist, even if negative. Decision about claims in the report.

---

## P5: System integration (weeks 14-19)

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P5-D1 | End-to-end live pipeline with webcam + phone IMU stream | D (+C) | L | Live demo | Runs 10 minutes without crash |
| P5-D2 | ONNX export of vision and fusion models, ONNX Runtime inference | D (+C) | L | `inference/export.py` | Output matches PyTorch within tolerance |
| P5-D3 | Latency and memory benchmarks (A9) | D | M | `A9.md` | Table on laptop CPU and one phone or edge device |
| P5-D4 | Degraded-mode behaviour in the live app (face lost, IMU lost) | D (+C) | M | Demo clips | Status messages shown correctly |
| P5-C1 | Quantization or pruning if latency misses the budget | C | L | Optimized model | Latency within budget or documented |
| P5-A1 | Robust face tracking in cabin conditions (glasses, low light) | A | M | Improved `vision.py` | Face-visible rate improved on own data |
| P5-B1 | Live IMU ingestion and online feature computation with ring buffers | B | L | `features/motion.py` online mode | Online features equal offline within tolerance |
| P5-ALL | Demo script and rehearsal | All | M | Demo checklist | Two successful rehearsals |

**Phase gate:** live demo on laptop, and recorded fallback demo.

---

## P6: Evaluation and own data (weeks 17-22)

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P6-D1 | Freeze own dataset v1.0, dataset card | D | M | `data/README.md` | Card reviewed |
| P6-C1 | Final runs for all ablations with 3+ seeds on frozen splits | C | L | `experiments/` | Tables regenerate from `make reproduce` |
| P6-C2 | `scripts/make_all_figures.py` for every plot in the report | C | M | Figures | One command regenerates |
| P6-A1 | Error analysis for vision classes (failure cases with consent-safe images) | A | M | `experiments/error_analysis_vision.md` | Top failure categories documented |
| P6-B1 | Error analysis for motion and events | B | M | `experiments/error_analysis_motion.md` | Documented |
| P6-D2 | User study with 3-5 volunteers viewing alerts and explanations (clarity score 1-5) | D | M | Survey results | Summary table |
| P6-ALL | Threats to validity section (`04_experiment_protocol.md` section 9) | All | S | Text | Reviewed by the guide |

---

## P7: Reports and polish (weeks 21-24)

| ID | Task | Owner | Effort | Deliverable | Done when |
|---|---|---|---|---|---|
| P7-D1 | Report assembly, formatting, references check | D | L | Final report PDF | Guide approves |
| P7-C1 | Results, experiment, methodology chapters | C | L | Chapter drafts | Reviewed by all |
| P7-A1 | Literature survey, vision sections | A | M | Chapter drafts | Reviewed |
| P7-B1 | Motion, data and preprocessing sections | B | M | Chapter drafts | Reviewed |
| P7-ALL | Final presentation, demo video, repo cleanup, README | All | M | Slides, video, tidy repo | Dry run done |

---

## Cross-phase recurring tasks

| Cadence | Task | Owner |
|---|---|---|
| Weekly | 20-minute sync, update board, update risk register | D facilitates |
| Weekly | One paper read, 5-line note | Everyone |
| Weekly | Each person reviews at least one PR | Everyone |
| Every two weeks | Short demo of the current state to the guide | Rotating |
| Every phase end | Merge `dev` into `main`, tag `vX.Y`, update the plan doc | D |
