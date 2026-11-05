# 03. Data Contract

**This is the most important document for parallel work.** Workstreams A and B produce data in this format; C and D consume it. Changing anything here after it is frozen needs a PR that updates this file and notifies all four members.

Status: DRAFT v0.1. **Freeze target: end of week 1** (decisions D1-D3 in `00_PROJECT_PLAN.md`).
Version of this contract: `contract_version = "0.1"` (stored in every output file; bump on any change).

---

## 1. Principles

1. One **common clock** for all modalities: **10 Hz** (one step = 0.1 s).
2. Every modality arrives as `[T, D]` features plus an **availability mask** `[T]`.
3. Missing data is never silently zero-filled. It is NaN in storage and masked in the model.
4. Raw datasets are converted once into **per-trip parquet files**. Windows are cut on the fly by the dataset class, never stored.
5. Splits are by **driver**, never by window (ADR-0005).

---

## 2. Terminology

| Term | Meaning |
|---|---|
| Trip | One continuous recording session of one driver |
| Step | One tick of the 10 Hz common clock |
| Window | `T = 50` consecutive steps (5 s) cut with stride 10 steps (1 s). Proposal D1 |
| Context feature | A slow feature over a longer span, for example PERCLOS over 60 s, computed causally and stored per step |
| Mask | Boolean, True means the modality has valid data at this step |

---

## 3. Source datasets (verify on download)

Fill in the "Verified" column when someone has downloaded and inspected the data. Until verified, treat the "Expected" column as an assumption, not a fact.

### 3.1 UAH-DriveSet (owner: Tanishq)

| Item | Expected | Verified |
|---|---|---|
| Drivers / vehicles | Small number of drivers (about 6) | [ ] |
| Behaviours | normal, drowsy, aggressive (trip-level styles) | [ ] |
| Road types | motorway, secondary road | [ ] |
| Raw accelerometer rate | about 10 Hz, with filtered acceleration and roll/pitch/yaw | [ ] |
| GPS rate | about 1 Hz | [ ] |
| Video | Road-ahead camera, not the driver's face | [ ] |
| Derived files | Lane detection, vehicle detection, map data, inertial events, semantic labels | [ ] |
| Distraction label | **None expected** | [ ] |
| License / citation terms | Read before use | [ ] |

### 3.2 DMD (owner: Anuja)

| Item | Expected | Verified |
|---|---|---|
| Subjects | Several tens of drivers | [ ] |
| Views | Face camera, plus body and hands views | [ ] |
| Modalities | RGB, with depth and IR in some parts | [ ] |
| Labels | Distraction behaviours, drowsiness related behaviours, gaze zones | [ ] |
| Real vs simulated driving | Both, check which subsets | [ ] |
| Vehicle motion signals | **None expected** | [ ] |
| License / access terms | Register and accept terms before download | [ ] |

### 3.3 Own recordings (owner: Gautam logistics, all members record)

Phone front camera on the driver, phone IMU and GPS, same trip, same clock. Protocol: `06_data_collection_protocol.md`.

### 3.4 Consequence: the paired-data problem

If the expectations above hold, **no public dataset has driver-face video and phone IMU/GPS for the same trip.** UAH gives motion (and a road view), DMD gives driver vision. Fusion therefore needs one of: own paired data (primary), or bootstrapping from public sets (ADR-0003). Every dataset loader must tag samples with `pairing = "paired" | "vision_only" | "motion_only"` so the training code can handle each case.

---

## 4. Feature schema

### 4.1 Vision features `X_v` (Dv = 16, proposal)

| Idx | Name | Unit | Definition / notes |
|---|---|---|---|
| 0 | `ear_left` | ratio | Eye aspect ratio, left eye, from landmarks |
| 1 | `ear_right` | ratio | Right eye |
| 2 | `ear_mean` | ratio | Mean of the two, primary eye-closure signal |
| 3 | `mar` | ratio | Mouth aspect ratio (yawn indicator) |
| 4 | `blink` | {0,1} | Blink event flag at this step |
| 5 | `eye_closed` | {0,1} | EAR below the driver-calibrated threshold |
| 6 | `perclos_30s` | fraction | Fraction of last 30 s with eyes closed (causal) |
| 7 | `perclos_60s` | fraction | Same over last 60 s (causal) |
| 8 | `head_yaw` | deg | Head pose, positive = driver's right |
| 9 | `head_pitch` | deg | Positive = looking up |
| 10 | `head_roll` | deg | |
| 11 | `gaze_yaw` | deg | Gaze direction proxy (iris based if available, else from head pose) |
| 12 | `gaze_pitch` | deg | |
| 13 | `off_road_gaze` | {0,1} | Gaze outside the forward zone beyond a threshold |
| 14 | `face_conf` | [0,1] | Landmark detection confidence |
| 15 | `face_visible` | {0,1} | 1 if a face was detected at this step |

Mask `m_v[t] = face_visible[t] == 1 and face_conf[t] >= 0.5`.
Frame rate for extraction: 15-30 fps. Frames inside one 0.1 s step are aggregated (mean for continuous, max for flags).

Optional second vision input: **clip tensor** `[3, 16, 112, 112]` per second for the 3D-CNN path, stored as a path + frame index, not as pixels.

### 4.2 Motion features `X_m` (Dm = 14, proposal)

| Idx | Name | Unit | Definition / notes |
|---|---|---|---|
| 0 | `acc_long` | m/s^2 | Longitudinal acceleration, gravity removed, vehicle-aligned |
| 1 | `acc_lat` | m/s^2 | Lateral acceleration |
| 2 | `acc_vert` | m/s^2 | Vertical |
| 3 | `jerk_long` | m/s^3 | Derivative of `acc_long` |
| 4 | `jerk_lat` | m/s^3 | Derivative of `acc_lat` |
| 5 | `yaw_rate` | deg/s | From gyro if present, else derivative of yaw angle |
| 6 | `roll` | deg | Phone attitude |
| 7 | `pitch` | deg | Phone attitude |
| 8 | `speed` | m/s | GPS speed (interpolated to 10 Hz) |
| 9 | `speed_var_5s` | (m/s)^2 | Rolling variance over 5 s |
| 10 | `heading_rate` | deg/s | From GPS heading |
| 11 | `lane_offset` | m | Lateral offset from lane center. **Only where lane detection exists** (UAH derived files); NaN elsewhere |
| 12 | `lateral_drift_5s` | m | Rolling std of `lane_offset`, or proxy from `acc_lat` integral for own data |
| 13 | `gps_valid` | {0,1} | 1 if a GPS fix is current |

Mask `m_m[t]` = IMU sample present within the last 0.2 s.
Features 11 and 12 can be NaN for a modality-available step; the model gets a per-feature NaN indicator through `nan_mask` (see 6.2).

Axis convention: x forward, y left, z up (right-handed), after a one-time orientation alignment using the first stationary-to-driving transition. Alignment method is owned by workstream B and documented in `features/motion.py`.

### 4.3 Context features (shared, optional)

`speed_bucket` (stopped / urban / highway) and `time_of_day_bucket` are derived from motion and timestamps, passed to the forecaster only.

---

## 5. Label map (proposal D2)

### 5.1 Window-level class labels `y_cls` (K = 4)

| ID | Class | Main sources |
|---|---|---|
| 0 | `normal` | UAH normal trips, DMD safe driving, own normal |
| 1 | `aggressive` | UAH aggressive trips, own harsh events |
| 2 | `drowsy` | UAH drowsy trips, DMD eye-closure/yawning/micro-sleep segments, own posed micro-sleep |
| 3 | `distracted` | DMD distraction segments (texting, phone, reaching, etc.), own posed distraction |

Mapping tables from each dataset's original labels to these IDs live in `configs/label_map_<dataset>.yaml` and are reviewed by the team (not hard-coded in loaders).

### 5.2 Event-level labels `events` (for the forecaster and event metrics)

Per trip a list of `(onset_s, offset_s, type)`, `type` in:

| Type | Definition | Source |
|---|---|---|
| `harsh_accel` | Longitudinal acceleration above a threshold | UAH inertial events / own detector |
| `harsh_brake` | Longitudinal deceleration below a threshold | same |
| `sharp_turn` | Lateral acceleration above a threshold | same |
| `lane_weave` | Repeated lateral drift crossing a threshold | UAH lane detection / own heuristic |
| `micro_sleep` | Eyes closed for >= 1.0 s (and < 3 s) | DMD labels / own annotation |
| `distraction_episode` | Off-road gaze or manual distraction beyond a threshold | DMD labels / own annotation |

Thresholds are config values (`configs/events.yaml`), initial values chosen with the guide and from literature, and then fixed before any model evaluation.

### 5.3 Known label limitations (must appear in the report)

- UAH behaviour is a **trip-level driving style**, so every window in an "aggressive" trip is labeled aggressive even when the driver is momentarily calm. Event-level labels come from the derived event lists and must not be mixed up with trip-level class labels.
- Drowsy driving in public sets is performed or simulated, not naturally induced.
- DMD and UAH have different drivers and contexts, so cross-dataset results reflect domain shift as well as method quality.

---

## 6. File formats

### 6.1 Directory layout

```
data/processed/<dataset>/
├── manifest.csv                 # one row per trip
├── <trip_id>.parquet            # features at 10 Hz
├── <trip_id>_events.json        # event list (5.2)
└── <trip_id>_meta.json
```

`<dataset>` in {`uah`, `dmd`, `own`}. `trip_id` format: `<dataset>_<driverid>_<yyyymmdd>_<nn>`, for example `own_d03_20261120_02`.

### 6.2 Parquet columns

One row per 0.1 s step.

| Column | Type | Notes |
|---|---|---|
| `t` | float32 | Seconds since trip start on the common clock |
| `v_<name>` for each of the 16 vision features | float32 | NaN where unavailable |
| `m_<name>` for each of the 14 motion features | float32 | NaN where unavailable |
| `mask_v` | bool | Vision available at this step |
| `mask_m` | bool | Motion available at this step |
| `y_cls` | int8 | -1 if unlabeled |
| `contract_version` | str | Constant column |

Per-feature NaN indicators (`nan_mask`) are derived at load time from NaNs, not stored.

### 6.3 `manifest.csv`

| Column | Notes |
|---|---|
| `trip_id`, `dataset`, `driver_id` | `driver_id` is an anonymous token like `d03` |
| `pairing` | `paired`, `vision_only`, `motion_only` |
| `duration_s`, `n_steps` | |
| `behavior_tag` | Trip-level tag from the source dataset if any |
| `road_type`, `vehicle_id` | If known |
| `split` | `train`, `val`, `test` (assigned by script, see `04_experiment_protocol.md`) |
| `source_version` | Dataset version or recording batch |
| `contract_version` | |

### 6.4 `<trip_id>_meta.json`

Sampling rates used, filter settings, calibration constants (EAR threshold, orientation alignment matrix), extractor versions (git hash), and a free-text `notes` field.

---

## 7. Tensor shapes at the model boundary

A batch from `driverisk.data.datasets.WindowDataset`:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `x_v` | `[B, 50, 16]` | float32 | Vision features, NaN replaced by 0 and flagged in `nan_v` |
| `x_m` | `[B, 50, 14]` | float32 | Motion features, same treatment |
| `nan_v` | `[B, 50, 16]` | bool | True where original was NaN |
| `nan_m` | `[B, 50, 14]` | bool | |
| `mask_v` | `[B, 50]` | bool | Modality availability |
| `mask_m` | `[B, 50]` | bool | |
| `y_cls` | `[B]` | int64 | Window label (majority vote; -1 ignored) |
| `y_risk` | `[B, 3]` | float32 | Event within H = 2, 3, 5 s (0/1), -1 if unknown |
| `driver_id` | `[B]` | str / int | For personalization |
| `trip_id`, `t_start` | `[B]` | | For traceability |
| `clip` (optional) | `[B, 5, 3, 16, 112, 112]` | float32 | One 16-frame clip per second inside the window; absent unless enabled |

Normalization: per-feature mean/std computed on the **training split only**, saved in `configs/norm_stats.json`, applied in the dataset class.

---

## 8. Synchronization rules

1. Each source has its own clock. Convert all to seconds since trip start.
2. Estimate offset between sources with a **sync event** recorded at the start of own trips (phone tap or sharp shake visible in IMU plus a hand-wave or blink pattern visible on camera). Residual offset target: below 100 ms.
3. Resample with linear interpolation for continuous signals, nearest neighbor for flags.
4. If drift between clocks exceeds 50 ms over a trip, apply linear drift correction using a second sync event at the end (collection protocol requires one).
5. Every trip meta file stores the estimated offset and drift.
6. For public datasets where sync is given by the authors, keep their alignment and record that fact.

---

## 9. Stub data for week 2

Each workstream must publish stubs so others can build in parallel:

| Workstream | Stub | Location |
|---|---|---|
| A | `make_stub_vision.py` creating 5 synthetic trips of vision features with random masks | `tests/fixtures/` |
| B | `make_stub_motion.py` same for motion | `tests/fixtures/` |
| C | Fusion model forward pass on stub batches, loss decreases on a toy label rule | `tests/` |
| D | FastAPI endpoints returning canned alerts matching `05_api_contract.md` | `serving/` |

A stub trip uses the full parquet schema above, and a CI test validates the schema.

---

## 10. Validation checks (run by CI on every processed trip)

- Required columns present, dtypes correct, `contract_version` matches.
- `t` strictly increasing with step 0.1 s (tolerance 1e-3).
- Feature values within physical ranges (for example `ear_mean` in [0, 1], `face_conf` in [0, 1]).
- No all-NaN feature column unless declared in meta (`lane_offset` is allowed).
- `y_cls` in {-1, 0, 1, 2, 3}.
- `manifest.csv` has one row per parquet and `driver_id` is never missing.
- No driver appears in more than one split.

---

## 11. Change log

| Version | Date | Change | By |
|---|---|---|---|
| 0.1 | 2026-10-18 | Initial draft | Maruf |
