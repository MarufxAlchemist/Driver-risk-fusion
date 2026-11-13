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
| 5 | `eye_closed` | {0,1} | EAR below the eye-closure threshold: a global default from config until a per-driver calibration exists (P4); see rules below |
| 6 | `perclos_30s` | fraction | Fraction of last 30 s with eyes closed (causal); NaN until 10 s of history exist |
| 7 | `perclos_60s` | fraction | Same over last 60 s (causal); same NaN rule |
| 8 | `head_yaw` | deg | Head pose, positive = driver's right |
| 9 | `head_pitch` | deg | Positive = looking up |
| 10 | `head_roll` | deg | |
| 11 | `gaze_yaw` | deg | Gaze direction proxy (iris based if available, else from head pose) |
| 12 | `gaze_pitch` | deg | |
| 13 | `off_road_gaze` | {0,1} | Gaze outside the forward zone beyond a threshold |
| 14 | `face_conf` | [0,1] | Landmark detection confidence |
| 15 | `face_visible` | {0,1} | 1 if a face was detected at this step |

Mask `mask_v[t] = face_visible[t] == 1 and face_conf[t] >= 0.5`. The validator checks this rule on every row.
Frame rate for extraction: 15-30 fps. Frames inside one 0.1 s step are aggregated (mean for continuous, max for flags). All 16 vision features are **per 0.1 s step**, on the common clock; the slow ones (PERCLOS) are causal context values stored at every step.

Rules for undefined values (added 2026-11-13, assumptions A3 and A4 in `PROGRESS.md`):

- **No face detected at a step:** `face_visible = 0`, `face_conf = 0`, and the other 14 vision features are NaN. The mask is then False.
- **PERCLOS history:** `perclos_30s` and `perclos_60s` use the history available so far once at least 10 s exist (so during the first 30 or 60 s the window is shorter than the name says), and are NaN before that. The stub generator follows this; the validator does not enforce it, because the extractor does not exist yet.
- **`eye_closed` threshold:** a global default EAR threshold from `configs/` is used until per-driver calibration (P4-A1). DMD has no calibration phase, so it always uses the default. The threshold used and where it came from go into the trip meta as `ear_threshold` and `ear_threshold_source` (`global_default` or `driver_calibrated`). The validator does not require these keys yet.

Optional second vision input: **clip tensor** `[3, 16, 112, 112]` per second for the 3D-CNN path, stored as a path + frame index, not as pixels. Its embeddings do **not** enter `X_v`: they travel as a separate tensor `e_clip [B, 5, Dc]` (one embedding per second of a 50-step window), which the fusion model repeats over the 10 steps of its second. `X_v` therefore stays at 16 features whatever the vision option (F7, ADR-0002).

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

Mask `mask_m[t]` = IMU sample present within the last 0.2 s.
Features 11 and 12 can be NaN for a modality-available step; the model gets a per-feature NaN indicator through `nan_mask` (see 6.2).

Naming: the symbols for masks are always `mask_v` and `mask_m`. The prefix `m_` appears only in parquet column names and means "motion feature" (`m_roll`); never write `m_v` or `m_m` for a mask.

Axis convention: x forward, y left, z up (right-handed), after a one-time orientation alignment using the first stationary-to-driving transition. Alignment method is owned by workstream B and documented in `features/motion.py`.

### 4.3 Context features (shared, optional)

`speed_bucket` (stopped / urban / highway) and `time_of_day_bucket` are derived from motion and timestamps, passed to the forecaster only.

### 4.4 UAH-DriveSet conversion rules (F25; from the UAH paper, R11, not yet checked on files)

The loader must convert UAH to the units and axes of section 4.2. These rules come from reading the paper; the "Verified" column of 3.1 stays unticked until `scripts/inspect_uah.py` has been run on the real download and every rule below confirmed or corrected.

| Item | UAH raw | Contract | Rule |
|---|---|---|---|
| Acceleration unit | G | m/s^2 | multiply by 9.80665 |
| Axes | Y lateral, Z longitudinal | x forward, y left, z up | `acc_long` from Z, `acc_lat` from Y; the sign of each is checked against braking and turning events in the data |
| Speed | GPS speed in km/h | m/s | divide by 3.6 |
| Yaw rate | no gyro column, only roll/pitch/yaw angles | deg/s | `yaw_rate = d(unwrap(yaw))/dt` |
| Inertial events | logged only above 50 km/h | events list | events are valid only where speed > 50 km/h; at lower speed their absence means "not logged", not "none happened" (affects FA/h, see `04`) |

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

Per trip a JSON list of objects `{ "onset_s": float, "offset_s": float, "type": str, "uncertain": bool (optional, default false), "source": str (optional) }`, `type` in:

| Type | Definition | Source |
|---|---|---|
| `harsh_accel` | Longitudinal acceleration above a threshold | UAH inertial events / own detector |
| `harsh_brake` | Longitudinal deceleration below a threshold | same |
| `sharp_turn` | Lateral acceleration above a threshold | same |
| `lane_weave` | Repeated lateral drift crossing a threshold | UAH lane detection / own heuristic |
| `micro_sleep` | Eyes closed for >= 1.0 s (and < 3 s) | DMD labels / own annotation |
| `distraction_episode` | Off-road gaze or manual distraction beyond a threshold | DMD labels / own annotation |

Thresholds are config values (`configs/events.yaml`), initial values chosen with the guide and from literature, and then fixed before any model evaluation.

`uncertain: true` is set by an annotator who is not sure (`06_data_collection_protocol.md` section 7). Uncertain events are excluded from headline metrics (see 7.1). `source` records where the event came from (for example `uah_inertial`, `dmd_label`, `own_annotation`, `own_detector`). The validator checks the three required keys, the type name and that the interval lies inside the trip; it does not read `source`.

Event alert names in the API are these same names (ADR-0007, `05_api_contract.md` section 3.4).

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
| `m_<name>` for each of the 14 motion features | float32 | NaN where unavailable. The `m_` prefix means "motion", not "mask" |
| `mask_v` | bool | Vision available at this step |
| `mask_m` | bool | Motion available at this step |
| `y_cls` | int8 | -1 if unlabeled |
| `contract_version` | str | Constant column |

Per-feature NaN indicators (`nan_mask`) are derived at load time from NaNs, not stored.

### 6.3 `manifest.csv`

| Column | Notes |
|---|---|
| `trip_id`, `dataset`, `driver_id` | `driver_id` is an anonymous token like `d03`, unique only inside its dataset. The **global driver key** is `<dataset>:<driver_id>` (for example `uah:d03`); splits, the leakage check and personalization use it (ADR-0007, `schema.driver_key`) |
| `pairing` | `paired`, `vision_only`, `motion_only` |
| `duration_s`, `n_steps` | |
| `behavior_tag` | Trip-level tag from the source dataset if any |
| `road_type`, `vehicle_id` | If known |
| `split` | `train`, `val`, `test` (assigned by script, see `04_experiment_protocol.md`) |
| `source_version` | Dataset version or recording batch |
| `contract_version` | |
| `quality` | `ok` or `low` (F20, **planned**: not yet in `schema.MANIFEST_COLUMNS` or the validator). Own trips that failed a protocol check (`06` section 9) are `low` and are left out of headline results |

### 6.4 `<trip_id>_meta.json`

Sampling rates used, filter settings, calibration constants (EAR threshold, orientation alignment matrix), extractor versions (git hash), and a free-text `notes` field.

Keys the validator requires today: `trip_id`, `contract_version`, `sampling_rates`, `extractor_versions`, `notes`. Optional keys it knows:

| Key | Meaning |
|---|---|
| `all_nan_columns` | List of parquet columns that are legitimately all NaN in this trip (see section 10). Names must be real feature columns |
| `events_annotated` | `false` if nobody annotated events for this trip (default `true`). Event-risk targets are then unknown (-1), see 7.1 |

Keys planned for own data (F20, **not yet validated**): `posed_segments` and `parked_segments`, lists of `[start_s, end_s]`. Posed segments (an actor imitating drowsiness or distraction in a parked car, `06` section 4.2) are kept for training the classifier, but are excluded from false-alarms-per-hour and from speed-context training, because speed 0 would otherwise teach the model that parked means risky. Also planned: `ear_threshold` and `ear_threshold_source` (section 4.1).

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
| `y_cls` | `[B]` | int64 | Window label, rule in 7.1; -1 = no label |
| `y_risk` | `[B, 3]` | float32 | Event onset within H = 2, 3, 5 s after the window (0/1), -1 if unknown; rule in 7.1 |
| `driver_id` | `[B]` | str | The global driver key `<dataset>:<driver_id>`, for personalization |
| `trip_id`, `t_start` | `[B]` | | For traceability |
| `clip` (optional) | `[B, 5, 3, 16, 112, 112]` | float32 | One 16-frame clip per second inside the window; absent unless enabled |

Normalization: per-feature mean/std computed on the **training split only**, saved in `configs/norm_stats.json`, applied in the dataset class.

### 7.1 How window labels are derived (F10; assumptions A5 and A7)

Let `t_end` be the time of the last step of a window.

- **`y_cls`:** the majority class among the *labelled* steps of the window (steps with `y_cls = -1` do not vote). The result is -1 if fewer than 50% of the window's steps are labelled, or if two or more classes tie for the majority.
- **`y_risk[h]`** for `h` in {2, 3, 5}:
  - 1 if a certain (not `uncertain`) event has its **onset** in `(t_end, t_end + h]`; otherwise 0;
  - -1 (unknown; any loss and any metric must ignore it) if the trip has no event annotation (`events_annotated = false`), if `t_end + h` is past the end of the trip, or if an `uncertain` event has its onset inside `(t_end, t_end + h]`.
  - All event types count. A per-type target is a later extension.
  - An event already running at `t_end` does not make `y_risk` 1; only a new onset does.

These rules are implemented in `driverisk.data.datasets` (`window_label`, `risk_targets`) and tested; change both together.

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
- Every all-NaN feature column must be listed in the trip's meta `all_nan_columns`. Expected cases: all 16 vision columns in a `motion_only` trip; all 14 motion columns in a `vision_only` trip; `lane_offset` (and `lateral_drift_5s` where it cannot be computed) in datasets without lane detection, which is everything except UAH. An all-NaN column that is not declared is an error.
- `mask_v` equals `face_visible == 1 and face_conf >= 0.5` on every row.
- Flag columns contain only 0 or 1, and no feature contains `inf`.
- `y_cls` in {-1, 0, 1, 2, 3}.
- Every event has `onset_s`, `offset_s`, a known `type`, and lies inside the trip.
- `manifest.csv` has one row per parquet and `driver_id` is never missing.
- No driver appears in more than one split.

---

## 11. Change log

| Version | Date | Change | By |
|---|---|---|---|
| 0.1 | 2026-10-18 | Initial draft | Maruf |
| 0.1 (edited, no bump) | 2026-11-13 | Documented what `schema.py`, the validator and `WindowDataset` already do: `mask_v`/`mask_m` naming, per-step vision features and separate `e_clip`, no-face and PERCLOS rules, all-NaN declaration via meta, `uncertain` events, label derivation (7.1), global driver key, UAH conversion rules (4.4). Planned but not implemented: manifest `quality`, meta `posed_segments`/`parked_segments`/`ear_threshold*`. Version stays 0.1 until the freeze (P0-C2), which waits for the UAH and DMD inspection | Claude, approved by Maruf |
