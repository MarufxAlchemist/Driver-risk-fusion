# 11. Implementation Analysis (Step 1, before any code)

Status: DRAFT for Maruf's review. Author: Claude (lead-engineer role), 2026-11-05.
Scope: everything in `Docs/` (00-10, five ADRs, ADR index, CONTRIBUTING, PR template), both READMEs, and the eight PDFs in `Docs/Research-papers/`.

How I read the papers: R1, R4, R5, R8, R11 (UAH) and Ultra-Fusion were read in full (excluding reference lists). For R2 and R3 I read the whole body; in R3 the radar/LiDAR physics and multi-target-tracking parts (§II-A/C/E, §IV) were skimmed because they are outside our scope. **Not available in the repo, so not read:** R6 (Tan), R7 (Kuang), R9 (Chen, the "closest prior work") and R10 (DMD paper). Nothing below cites a number from a paper I have not opened.

Notation: `03 §4.1` means `03_data_contract.md` section 4.1. Proposed fixes are numbered **F1-F27** in section 7. Section 2 refers to them.

---

## 1. My understanding of the system (under 300 words)

The project builds a driver-risk monitor that runs on a laptop or phone in the car. A driver-facing camera gives 16 hand-crafted vision features per 0.1 s step (eye aspect ratio, PERCLOS, head pose, gaze proxy, face confidence). The phone IMU and GPS give 14 motion features (acceleration, jerk, yaw rate, speed variance, lane drift). Both are resampled to a 10 Hz common clock, each with an availability mask, and cut into 5 s windows with a 1 s stride. An optional 3D-CNN or 2D-CNN+LSTM clip branch is evaluated, not assumed.

A cross-attention fusion network lets each stream query the other, replaces masked steps with learned missing tokens, and uses gates. It is trained with modality dropout so that losing the camera or the phone degrades performance gracefully instead of breaking it. On top of the fused embedding sit three stages: an LSTM autoencoder trained on normal driving (an unsupervised abnormality score), a 4-class window classifier (normal / aggressive / drowsy / distracted), and a forecaster giving the probability that an abnormal event starts within 2, 3 or 5 s. A per-driver baseline turns scores into personal z-scores, shrunk toward global statistics while calibration data is scarce. An explanation module fills templates ("eyes closed 38% of the last minute, your usual 6%") from those deviations. Only features and alert JSON reach a FastAPI server and dashboard. Raw video never leaves the device.

The central scientific constraint is the **paired-data problem**. UAH-DriveSet has phone motion but no face video, and DMD has face video but no vehicle motion. Honest fusion results therefore need our own paired recordings (F1), helped by pretraining each branch on public data (F2). Synthetic class-matched pairing (F3) may only be used for robustness studies and must always be labelled "synthetic pairing". Every claim maps to ablations A1-A9, run with driver-disjoint splits, three seeds, and thresholds tuned only on validation data.

---

## 2. Inconsistencies and gaps

### 2.1 Repository and document set

| # | Issue | Where | Fix |
|---|---|---|---|
| G1 | Docs are flat in `Docs/` (capital D). Plan, README and master prompt expect `docs/`, `docs/decisions/`, `docs/reports/`, `.github/PULL_REQUEST_TEMPLATE.md`, root `CONTRIBUTING.md`. On Windows `Docs/` and `docs/` are the same folder but different paths to git, so mixing them breaks on Mac/Linux checkouts. | `00 §2`, `README.md` "Documentation index" | F1 |
| G2 | `README (1).md` is actually the **ADR index**, not a readme. | `Docs/README (1).md` | F1 |
| G3 | ADR-0006 "Technology stack and edge export path" is listed as Proposed, but no file exists. | ADR index table | F3 |
| G4 | `docs/reports/README.md` is referenced by the README, `07` P3 and the master prompt but does not exist. `docs/reports/consent_form_template.md` ("to be created in P0") also does not exist. | `README.md`, `06 §2`, `07` P3 | F4 |
| G5 | Eight PDFs (about 41 MB) are tracked in git. At least five are IEEE copies stamped "Authorized licensed use limited to ... Restrictions apply" (R2, R4, R5, R8, R11). Redistributing them in a public GitHub repo is likely a licence breach. R1 and R3 are CC-BY 4.0 and fine. Ultra-Fusion is an arXiv preprint with an unchecked licence. I could not check repo visibility (`gh` is not logged in). | `Docs/Research-papers/` | F2 |
| G6 | README says "License: to be decided", but `LICENSE` (MIT, Maruf Nadaf) already exists. ADR-0001 defaults to MIT. | `README.md`, `00 §7` D8 | F27 |
| G7 | README lists four members with workstreams, but the master prompt says one person builds all of it. `08` (reviews from a different workstream, backups, rotation) and `CONTRIBUTING §4` (approval from a different workstream, squash merge) cannot be followed as written. | `08 §1,3`, `CONTRIBUTING §4` | F23 |

### 2.2 Feature schema, counts and names (16 vision, 14 motion)

I checked counts everywhere they appear. They match: `03 §4.1` lists indices 0-15 (16), `03 §4.2` lists 0-13 (14), `03 §7` shapes `[B,50,16]` and `[B,50,14]`, ADR-0002 `X_v [T,16]`, `X_m [T,14]`. The `05 §3.3` example vectors have 16 and 14 entries. The `05 §3.4` z-scores are arithmetically correct ((0.38-0.06)/0.03 = 10.7; (-14+2)/4 = -3.0). The problems are in naming and semantics:

| # | Issue | Where | Fix |
|---|---|---|---|
| G8 | In the architecture, `m_v` and `m_m` mean **masks** ("`X_v [T, Dv]`, `m_v [T]`"). In parquet, the `m_` prefix means **motion** ("`m_<name>` for each of the 14 motion features"), and masks are `mask_v`/`mask_m`. A column like `m_roll` next to the symbol `m_v` invites bugs. | `02 §3` M3/M4/M7, `02 §4.3`; `03 §6.2` | F6 |
| G9 | 02 says vision features are produced "per 1 s step". The contract clock is 0.1 s per step. | `02 §4.1` table header "Output per 1 s step" | F7 |
| G10 | The clip path produces "Embedding `[Dc]` per step" (M5), but the batch has one clip **per second** (`clip [B, 5, 3, 16, 112, 112]`, 5 clips for 50 steps). How 5 clip embeddings align with 50 steps is not specified. ADR-0002 says the embedding "can be appended to `X_v`", which would silently change Dv=16. | `02 §3` M5, `02 §4.1`, `03 §7`, ADR-0002 | F7 |
| G11 | The vision mask requires `face_visible==1 and face_conf>=0.5`. When no face is found, the values of `face_conf` and the other 14 vision features are undefined (NaN or 0?). 02 says "landmark confidence < threshold" without the number. | `03 §4.1`, `02 §6` | F9 |
| G12 | `eye_closed` uses "the driver-calibrated threshold". Calibration only arrives in P4 (P4-A1), DMD has no calibration phase, and the threshold source for P1 is unspecified. | `03 §4.1` idx 5, `07` P1-A2, P4-A1 | F9 |
| G13 | `perclos_30s`/`perclos_60s` are causal, but the value during the first 30/60 s of a trip is undefined. | `03 §4.1` idx 6-7 | F9 |
| G14 | The validation rule "No all-NaN feature column unless declared in meta (`lane_offset` is allowed)" fails for every `motion_only` trip (16 vision columns all NaN), every `vision_only` trip, and own data (`lane_offset` NaN, plus `lateral_drift_5s` uses a proxy). | `03 §10`, `03 §3.4` | F8 |
| G15 | UAH units and axes differ from the contract (from the UAH paper, R11 §III-A): acceleration is in **G**, Y is lateral and **Z is longitudinal** (contract: x forward, y left, z up, m/s²). GPS speed is in **km/h** (contract m/s). There is no gyro-rate column, only roll/pitch/yaw angles, so `yaw_rate` must come from the derivative of unwrapped yaw. None of this is documented as a loader rule. | `03 §4.2` | F25 |
| G16 | "Phone moved: random rotation of the IMU axes" is a raw-signal degradation, but the contract only stores post-alignment features. This simulator needs a raw-IMU hook before feature extraction. | `04 §3.4` | F24 |

### 2.3 Windows, stride, labels and IDs

| # | Issue | Where | Fix |
|---|---|---|---|
| G17 | Window and stride (50 steps, stride 10) are consistent across `00 §7` D1, `03 §2`, ADR-0002 and `10`. No issue. | n/a | n/a |
| G18 | `y_cls` "majority vote; -1 ignored" does not say what happens on ties, or when most steps are -1. | `03 §7` | F10 |
| G19 | `y_risk` "Event within H = 2, 3, 5 s ... -1 if unknown" does not define t (window end?), which event types count, or when the value is "unknown" (no annotation? t+H past trip end?). There is no `y_risk` column in parquet, so it must be derived from `_events.json`, whose schema is only a tuple `(onset_s, offset_s, type)`. | `03 §5.2`, `§7`, `02 §4.7` | F10 |
| G20 | `06 §7` asks annotators to mark events `uncertain` and exclude them from headline metrics, but the events schema has no such field. | `06 §7` vs `03 §5.2` | F10 |
| G21 | **Three event/alert taxonomies that do not match.** Classes: `normal, aggressive, drowsy, distracted` (`03 §5.1`). Events: `harsh_accel, harsh_brake, sharp_turn, lane_weave, micro_sleep, distraction_episode` (`03 §5.2`). API alert `type`: `aggressive_accel, harsh_brake, sharp_turn, lane_weave, drowsy, micro_sleep, distracted` (`05 §3.4`). So `aggressive_accel` ≠ `harsh_accel`; `distracted` ≠ `distraction_episode`; there is no alert type for class `aggressive`; and a `predictive` alert of type `drowsy` exists although the forecaster predicts *event onsets*, and `drowsy` is not an event. | `03 §5.1-5.2`, `05 §3.4` | F12 |
| G22 | Templates: no template for `sharp_turn`; a `yawning` template exists without a yawning event or alert type; the `05 §3.4` example summary adds "Camera is the main source." which is not in the `drowsy_perclos` template. | `05 §3.5` | F12 |
| G23 | ID formats: contract driver `d03`, trip `own_d03_20261120_02`; API driver `drv_a1b2`, trip `trp_77ef`. No mapping is defined. Driver IDs also collide across datasets (`d03` can exist in UAH, DMD and own), and splits and personalization key on driver. `driver_id` dtype is "str / int". | `03 §6.1, 6.3, 7`, `05 §1` | F11 |
| G24 | Own-data file naming `<driverid>_<yyyymmdd>_<nn>_{video,imu,gps}` lacks the dataset prefix used in `trip_id`. | `06 §5` step 8 vs `03 §6.1` | F11 |
| G25 | Trip quality: `06 §9` flags failed trips `quality = "low"`, but `manifest.csv` has no quality column. Posed segments (parked car, `06 §4.2`) are not marked anywhere in the contract, but must be excluded from FA/h and from speed context (see G33). | `06 §9`, `03 §6.3` | F20 |

### 2.4 Endpoints and schemas (05)

| # | Issue | Where | Fix |
|---|---|---|---|
| G26 | The scripted mock sequence ends with "recovered", but WS message types are only `status, alert, degraded, trip_ended`. | `05 §5` vs `§3.6` | F19 |
| G27 | Auth is the header `X-API-Key`, but browsers cannot set custom headers on a WebSocket, so the dashboard cannot authenticate `/stream` as specified. The OpenAPI snapshot test also does not cover WebSocket messages. | `05 §1`, `§3.6` | F19 |
| G28 | No request/response schema is given for `/drivers/{id}/baseline`, `/calibrate`, `/steps` (response), `POST /alerts` (response), `/end`, `/replay`, `/health`, or `GET /trips` (pagination). Severity mapping and `confidence` (vs `risk_score`) are undefined. | `05 §2-3` | F19 |
| G29 | Mock server location: `scripts/mock_server.py` (`05 §5`, `07` P0-D7) vs "FastAPI endpoints ... `serving/`" (`03 §9`). The master prompt puts schemas in `serving/schemas.py`. | `03 §9`, `05 §5` | F19 |
| G30 | `05 §1` says "this file must match" `schemas.py`, and the master prompt says Pydantic is the source of truth. The OpenAPI snapshot is at `docs/openapi_snapshot.json`, not under tests. | `05 §1` | F19 |

### 2.5 Personalization and calibration constants

| # | Issue | Where | Fix |
|---|---|---|---|
| G31 | Three unrelated definitions of "enough calibration": `n0 = 20` windows prior strength (`02 §4.6`); first `K = 2` normal trips (`04 §2.2`); `windows_required: 600` (`05 §3.1`). `n_min` (`02 §6`) is never given a value. | `02 §4.6, §6`, `04 §2.2`, `05 §3.1` | F13 |
| G32 | The symbol `K` means both the number of classes (K = 4) and the number of calibration trips (K = 2). "Same for sigma" in the shrinkage formula does not say whether variance or std is shrunk. | `03 §5.1`, `02 §4.5-4.6`, `04 §2.2` | F13 |
| G33 | Parked posed segments (speed 0) conflict with the speed context feature. R1 states that "if the vehicle is stopped, the determined possible driver distraction is irrelevant", so a speed-aware alert rule (A8) would suppress exactly the posed drowsy/distracted data we rely on. | `06 §4.2`, `03 §4.3`, `04` A8 | F20 |
| G34 | Safety gap: a personal z-score can desensitize alerts for a driver whose "normal" is already risky. R8 §VI-E warns that over-personalization "could compromise safety". No absolute floor is specified. | `02 §4.6` | F14 |
| G35 | UAH personalization feasibility (from R11 §II): each driver has 3 normal trips (D6 has 2: one motorway, one secondary). With K = 2 normal calibration trips, only 1 normal trip (0 for D6) is left to measure FA/h, and "chronologically later" depends on folder timestamps not yet inspected. | `04 §2.2` | F13 |

### 2.6 Metrics and protocol

| # | Issue | Where | Fix |
|---|---|---|---|
| G36 | Macro-F1 over K = 4 is undefined on DMD (no `aggressive`) and UAH (no `distracted`). Per-dataset class subsets are not stated. | `04 §3.1` | F15 |
| G37 | FA/h divides "alerts with no real abnormal event within `[t-2 s, t+H]`" by "hours of normal driving". The numerator counts alerts anywhere, the denominator only normal time. "Normal" is undefined for UAH aggressive/drowsy trips (trip-level labels). H = 0 for detected alerts. The alert-to-event matching rule (one-to-one? first alert?) used by recall and lead time is undefined. | `04 §3.2` | F15 |
| G38 | UAH logs inertial events **only above 50 km/h** (R11 §III-C, "if the speed is less than 50Km/h ... the events are not saved"). Real harsh events at low speed are unlabeled, so they would count as false alarms. | `03 §5.2` source column | F15, F25 |
| G39 | Leave-one-driver-out with early stopping "on validation macro-F1" does not say where the validation driver comes from (nested folds needed). `norm_stats.json` lives in `configs/`, but statistics differ per fold and per setting (F1/F2/F3), so one file cannot serve all runs. | `04 §1.3, §2, §6`, `03 §7` | F16 |
| G40 | A1 success criterion "by more than 1 std": whose std (M2's, the baseline's, pooled)? A3 has two different criteria (`04 §3.4`: "degrades more slowly than concatenation and never below the better single modality by >1 std" vs `04 §5` A3: "M2 curve above M1 for r ≥ 0.25"). | `04 §3.4, §5` | F17 |
| G41 | `make reproduce RUN=<run_id>` (re-run one config, `04 §8`, "created in P5") vs "Tables regenerate from `make reproduce`" (`07` P6-C1) vs master DoD "`make reproduce` regenerates the final tables and figures". These are different commands. | `04 §8`, `07` P6-C1 | F18 |
| G42 | `10` glossary: F1/F2/F3 (fusion settings) collide with "F1" (score) in tables such as "macro-F1 (F1 setting)". | `10`, `04 §7` | F17 |
| G43 | Time-offset perturbation is missing from the robustness sweep, although sync error is a top-5 risk (R5). Ultra-Fusion evaluates injected delays explicitly. | `04 §3.4` | F24 |

### 2.7 Phase tasks and dependencies (07)

| # | Issue | Where | Fix |
|---|---|---|---|
| G44 | The P0 phase gate needs "-> dashboard shell", and P0-D7 is "done when: Dashboard shell can display the scripted sequence", but the dashboard shell is task **P1-D3**. | `07` P0 gate, P0-D7, P1-D3 | F18 |
| G45 | P0-C2 freezes "contract v1.0", but the contract string is `"0.1"` and `05` examples send `"contract_version": "0.1"`. The versioning rule for the freeze is not stated. | `07` P0-C2, `03` header | F18 |
| G46 | No task produces `experiments/ablations/A7.md` (P2-A3 delivers "Run logs + ADR update"). `04 §5` requires a file per ablation. | `07` P2-A3, `04 §5` | F18 |
| G47 | P0-ALL is owned by "C" although its ID says ALL. It asks for notes on R9, whose PDF is not in the repo. | `07` P0-ALL | Human task H10 |
| G48 | `schema.py` (master prompt §3.2.3) is not mentioned in any doc. `07` names `data/datasets.py` and `data/validate.py`. The class path `driverisk.data.datasets.WindowDataset` (`03 §7`) is consistent with that. | `07` P0-C3 | F5 |
| G49 | Dependencies that 07 leaves implicit: P1-C1 (label maps) needs P0-A1/B1 verification; P1-D1 needs P0-D5 (app choice); P1-D2 needs P0-D6 (guide approval); P2-C2 needs P1-D2 pilot data; P4-C3 needs event JSONs (P1-B3, P2-D2). | `07` | Reflected in §4 |
| G50 | 02 lists module M1/M2 paths as `data/`, which is the git-ignored data folder; code lives in `src/driverisk/data/`. | `02 §3` | F6 |

### 2.8 Literature notes vs what the papers actually say

| # | Issue | Evidence | Fix |
|---|---|---|---|
| L1 | R5 (Kose 2019) is cited for "Why 3D CNN?". It is **not** a 3D-CNN: it is a 2D BN-Inception on 4-8 sparsely sampled frames with concatenated features. It supports "temporal modelling beats single frames" and is closer to our B5 baseline than to B4. | R5 §III | F21 |
| L2 | R5 is strong evidence for ADR-0005, but the notes do not use it: 99.10% on a split where the same drivers appear in train and test, vs 92.67-94% in subject-disjoint 5-fold CV (RGB). | R5 §IV-A/B, Table IV | F21 |
| L3 | R8 does **not** state that "most deployed systems use one fixed global threshold". It does support "open direction" (§VI-A: no open personalized datasets; thresholds and triggers for model updates are unexplored). It also warns about over-personalization (§VI-E). | R8 §VI | F21, F14 |
| L4 | R1 does not show that motion detects cognitive distraction. It says there is currently no non-intrusive way to measure mental state, and that driving-style signals give *indirect* evidence of distraction "compared to the usual driver parameters acquired before". That sentence actually supports our personalization idea. | R1 §IV-A/B | F21 |
| L5 | The R3 guide answer cites "ablation A3 vs A4/A5" (should be A2/A3). R3 does not discuss cross-attention, missing modalities or modality dropout. It covers fusion levels, time-space alignment and camera degradation (detection drops 31.1-60.4% under corruptions). | R3 §III, §V; `01 §3` R3 | F21 |
| L6 | **Prior camera + phone-IMU fusion exists.** R2 §III-B cites Streiffer et al. (face camera + mobile IMU, InceptionV3+LSTM, 87.02% on a private set), Omerustaoglu et al. (vision + vehicle sensors, +9 points to 85%), and Ou et al. (head pose + distraction + speed). The novelty must be framed as the *combination* (degradation-robust fusion + personal baseline + forecasting + explanations), which `01 §2` "GAP" already does. Do not claim "first camera+IMU fusion". | R2 §III-B-2b, Table XVIII | F21 |
| L7 | R2 reports Brain4Cars: driver-face video + speed + GPS, 10 drivers, 594 five-second clips, with a time-to-maneuver (TTM) metric. It is not phone IMU and not abnormal driving, so it does not resolve ADR-0003. ADR-0003 asks to "check once more in P0" for paired data, so it should be named and dismissed explicitly. | R2 §III-C | F21 |
| L8 | R4 supports "placement matters" (58.3-85.5% depending on camera and model, middle best). R2's conclusion says gaze accuracy was "not observed to clearly" depend on placement. Cite both and keep the claim modest. R4 is a single split and single run. | R4 Table II; R2 §IV | F21 |
| L9 | Ultra-Fusion (arXiv 2606.21223, June 2026, unreviewed SLAM preprint) is in the folder but not in R1-R11. Useful ideas: per-modality reliability scores with **hysteresis**, cross-correlation time-offset estimation, staged degradation tests. Cite it only for the robust-fusion principle, labelled as a preprint from another domain. | Ultra-Fusion §III-D/E | F21, F24 |
| L10 | Citation check: R8 is vol. 10, **no. 2**, pp. 1241-1262, DOI 10.1109/TIV.2024.3425647 ✓. R1, R2, R3, R4, R5 and R11 DOIs and pages match the PDFs ✓. | PDFs | n/a |

---

## 3. "Verify" assumptions and how each will be verified

All Expected values below remain assumptions until someone has the data on disk. The "Paper says" column comes from the dataset paper, not from inspection. I only have the paper for UAH (R11). The DMD paper (R10) is not in the repo.

### 3.1 UAH-DriveSet (`03 §3.1`)

| Item | Expected (contract) | Paper says (R11, Romera 2016) | How to verify (script `scripts/inspect_uah.py`, run after download) |
|---|---|---|---|
| Drivers / vehicles | about 6 | 6 drivers (D1-D6), 6 vehicles incl. one EV | Count driver folders; list behaviour/road per trip from folder names `Date-Distance-Driver-Behavior-Road` |
| Behaviours | normal, drowsy, aggressive | Same, **simulated** by instruction; per driver 3 motorway + 4 secondary trips (D6: 3 + 2) | Tabulate trips per driver × behaviour × road; expect about 40 trips, more than 500 min |
| Road types | motorway, secondary | Same | From folder names |
| Accelerometer rate | about 10 Hz, filtered acc + roll/pitch/yaw | 10 Hz (mean of 100 Hz), raw + Kalman-filtered X/Y/Z **in G**, roll/pitch/yaw in degrees; **Y lateral, Z longitudinal**; no gyro rates | Median and max Δt of `RAW_ACCELEROMETERS`; column count = 11; value ranges near ±1 G; sign of Z during a known braking |
| GPS rate | about 1 Hz | 1 Hz; speed **km/h**, course, difcourse | Median Δt of `RAW_GPS`; speed range |
| Video | road ahead | Second phone, road-facing | Open the first frame of 3 videos |
| Derived files | lane, vehicle, map, events, semantics | `PROC_LANE_DETECTION` (~30 Hz variable, X = offset from lane centre in m, detector state), `PROC_VEHICLE_DETECTION` (~10 Hz), `PROC_OPENSTREETMAP_DATA` (~1 Hz), `EVENTS_LIST_LANE_CHANGES`, `EVENTS_INERTIAL` (brake/turn/accel × 3 levels, **only above 50 km/h**), `SEMANTIC_FINAL/ONLINE` | Check all 9 files per trip; event counts per behaviour; fraction of time above 50 km/h |
| Distraction label | none | None (no face camera) | Confirm by file list |
| License / citation | read before use | Not stated in the paper | Human reads the dataset web page and readme (H1) |

Extra checks that matter for later phases: whether trip timestamps allow a chronological order per driver (needed for `04 §2.2`), and the lane-detector `state == 2` coverage (decides whether `lane_offset` is usable).

### 3.2 DMD (`03 §3.2`)

The paper is not in the repo, so I have no "paper says" column. Verification needs registration (H1). After download, `scripts/inspect_dmd.py` will:

| Item | Expected | How to verify |
|---|---|---|
| Subjects | several tens | Count unique subject IDs in annotation files |
| Views | face, body, hands | List video streams per session |
| Modalities | RGB, depth and IR in parts | List file types per stream |
| Labels | distraction, drowsiness-related, gaze zones | Parse the annotation files, dump the label vocabulary and interval counts per label |
| Real vs simulated | both | Read session metadata; tabulate per subset |
| Vehicle motion | none | Confirm no IMU/CAN files |
| Licence | registration + terms | Human reads and accepts (H1); record terms in `data/README.md` |

Extra checks: annotation frame rate vs video fps, and whether drowsiness labels include eye-closure intervals that are usable for `micro_sleep` (≥1 s and <3 s).

### 3.3 Other assumptions marked as unverified elsewhere

| Assumption | Where | Verification |
|---|---|---|
| Phone logging app exports timestamped CSV at a stable rate on Android and iOS | `00 §7` D6, `06 §3` | `scripts/check_phone_log.py` on a 5-minute test log: rate, jitter, gaps over 0.5 s, timestamp format (H4) |
| Tap-and-blink sync gives a residual offset under 100 ms | `03 §8`, `06 §5` | Parked-car test: detect both taps in IMU and in video, compare with the cross-correlation estimate (H4) |
| MediaPipe Face Landmarker runs at 15 fps or better on the laptop CPU within 40 ms per frame | `02 §5` | Benchmark script in P1 on the webcam |
| No public dataset pairs face video with phone IMU | ADR-0003 | Literature check: Brain4Cars (no IMU), Streiffer et al. dataset (availability unknown; read the original) |

---

## 4. Solo execution plan (single critical path)

Assumptions: Week 1 = week of 2026-11-02 (verify against your college calendar). "C" = my hours (code, tests, docs). "M" = your hours (review, decisions and human-only work). Tasks marked **NEW** are needed but not listed in 07. Order follows the stubs-first rule: nothing waits for real data while stub work is possible.

### P0 Foundations (weeks 1-3)

| # | Task | Depends on | C h | M h |
|---|---|---|---|---|
| 1 | Doc fixes you accept from §7, repo re-layout (F1), branch `dev` | your answer | 2 | 1 |
| 2 | P0-D2 scaffold: `pyproject.toml` (Py 3.11, pinned), Makefile + Windows shim, `.gitignore`, pre-commit, `data/README.md` | 1 | 4 | 1 |
| 3 | P0-D3 CI (ruff + pytest, ubuntu + windows) | 2 | 2 | 0.5 |
| 4 | NEW utils: `seed.py`, `paths.py`, YAML config loader | 2 | 2 | 0.5 |
| 5 | P0-C3a `data/schema.py` (single source of names/counts/labels) + `data/validate.py` + contract tests | 4 | 4 | 1 |
| 6 | P0-A2, P0-B2 stub generators (parquet, random masks, NaNs, events json) | 5 | 3 | 0.5 |
| 7 | P0-C3b `WindowDataset` + driver-disjoint split enforcement + tests | 6 | 5 | 1 |
| 8 | P0-C4 toy cross-attention fusion + tiny training loop (loss drops on toy rule) | 7 | 4 | 1 |
| 9 | P0-D7 Pydantic schemas, mock FastAPI + scripted WS sequence, OpenAPI snapshot test | 4 | 6 | 1 |
| 10 | NEW minimal dashboard shell (one static HTML page reading the WS) for the gate | 9 | 3 | 0.5 |
| 11 | NEW `make demo-stub` + integration test | 8, 9, 10 | 2 | 0.5 |
| 12 | NEW `docs/human_tasks/` pack (master prompt §5) | 1 | 5 | 1 |
| 13 | P0-C1 decisions D1-D8 → ADR-0006 (stack), ADR-0007 (taxonomy, IDs) | 1 | 2 | 2 |
| 14 | P0-A1, P0-B1 dataset download + inspection scripts, fill "Verified" | H1 | 4 | 6 |
| 15 | P0-D5 phone app test + checker script | H4 | 2 | 4 |
| 16 | P0-D6 guide meeting (one-pager, consent template) | 12 | 1 | 2 |
| 17 | P0-ALL R8/R9 notes (R9 PDF needed) | H10 | 2 | 1 |
| 18 | P0-C2 freeze contract v1.0 | 13, 14 | 2 | 1 |
| | **P0 total** | | **55** | **25.5** |

### P1 Data and features (weeks 3-7)

| # | Task | Depends on | C h | M h |
|---|---|---|---|---|
| 19 | P1-C3 normalization stats (per run/fold) + augmentations, on stubs | 7 | 4 | 0.5 |
| 20 | P1-D4 experiment logging (CSV + run folder; MLflow/W&B skipped unless you want it) | 4 | 3 | 0.5 |
| 21 | **Moved up from P2:** P2-C1 metrics library (macro-F1, FA/h, lead time, ECE, PR-AUC) tested on synthetic cases | 13 | 10 | 1.5 |
| 22 | P1-B2 motion features, tested on synthetic signals | 5 | 10 | 1.5 |
| 23 | P1-B3 event detector + `configs/events.yaml` | 22 | 5 | 1 |
| 24 | P1-A1 MediaPipe vision features (developed on your own webcam clips) | 5 | 12 | 2 |
| 25 | P1-A2 causal PERCLOS, blink, eye-closed with default threshold | 24 | 5 | 2 |
| 26 | P1-C1 label maps per dataset | 14 | 2 | 1 |
| 27 | P1-B1 UAH loader → parquet (unit/axis rules F25) | 14, 22, 26 | 10 | 1 |
| 28 | P1-A3 DMD loader → parquet | 14, 24, 26 | 12 | 1 |
| 29 | P1-C2 split script + frozen `splits.json` | 27, 28 | 4 | 0.5 |
| 30 | P1-D1 own-data processing: tap detection + cross-correlation sync, resample, parquet | 15, 22, 24 | 10 | 1 |
| 31 | P1-D2 pilot recordings, 2 drivers × 2 trips | 16 (approval) | 0 | 8 |
| 32 | P1-D3 dashboard: trip list, risk gauge, alert feed (mock) | 10 | 6 | 1 |
| 33 | Phase gate: one DataLoader over UAH + DMD + pilot, `experiments/data_report.md` | 27-31 | 2 | 0.5 |
| | **P1 total** | | **95** | **23** |

### P2 Baselines (weeks 7-10)

| # | Task | C h | M h |
|---|---|---|---|
| 34 | P2-C3 training loop, config system, checkpointing (extends #8) | 6 | 1 |
| 35 | P2-B1 B1 rules + B2 sensor-only LSTM on UAH (nested LODO × 3 seeds) | 6 | 1 |
| 36 | P2-B2 Stage 1 LSTM-AE + AUROC | 8 | 1 |
| 37 | P2-A1 B3 camera-only feature LSTM on DMD | 6 | 1 |
| 38 | P2-A2 B5 2D-CNN+LSTM on a DMD subset (GPU) | 10 | 2 |
| 39 | P2-A3 B4 3D-CNN + go/no-go + `A7.md` (GPU) | 10 | 2 |
| 40 | P2-C2 B6/B7 on pilot data and on F3 (labelled synthetic) | 6 | 1 |
| 41 | P2-D2 annotation CSV workflow + kappa / interval-IoU script | 6 | 8 |
| 42 | P2-D3 results summary generated by script | 2 | 0.5 |
| 43 | P2-D1 recording continues (human) | 0 | 12 |
| | **P2 total** | **60** | **29.5** |

### P3 Fusion (weeks 10-13)

| # | Task | C h | M h |
|---|---|---|---|
| 44 | P3-C1 full cross-attention fusion (masks, missing tokens, gates) + masked-stream tests | 10 | 1.5 |
| 45 | P3-A1 vision embedding export (V-feat, V-3D if go) | 4 | 0.5 |
| 46 | P3-A2, P3-B1 degradation simulators (incl. raw-IMU rotation, time offset) | 8 | 1 |
| 47 | P3-C2 modality dropout runs, p ∈ {0, 0.1, 0.15, 0.3} | 4 | 0.5 |
| 48 | P3-C3 ablations A1, A2 | 8 | 2 |
| 49 | P3-C4 ablation A3 robustness curves | 8 | 1 |
| 50 | P3-B2 ablation A6 (AE score as input) | 4 | 0.5 |
| 51 | P3-D1 real-time inference engine with replay | 10 | 1 |
| 52 | P3-D2 WebSocket integration with real alert objects | 5 | 1 |
| 53 | P3-D3 recording + dataset v0.5 card | 2 | 8 |
| | **P3 total** | **63** | **17** |

### P4 Personalization and prediction (weeks 13-17)

| # | Task | C h | M h |
|---|---|---|---|
| 54 | P4-C1 personal baseline (shrinkage, drift freeze, safety floor F14) | 8 | 1 |
| 55 | P4-C3 forecaster (label generation, model, PR-AUC per horizon) | 10 | 1.5 |
| 56 | P4-C4 alerting rule + debounce + A5 | 8 | 1.5 |
| 57 | P4-C2 ablation A4 per-driver tables | 6 | 1 |
| 58 | P4-C5 explanation module + `templates.yaml` | 6 | 1 |
| 59 | P4-A1 per-driver EAR calibration | 4 | 1 |
| 60 | P4-B1 context features + A8; P4-B2 per-driver motion normalization | 7 | 1 |
| 61 | P4-D1 dashboard baseline/"why" views; P4-D2 calibration flow | 12 | 1.5 |
| 62 | Personalization trips (human) | 0 | 10 |
| | **P4 total** | **61** | **19.5** |

### P5-P7 (weeks 17-24)

| Phase | Main tasks | C h | M h |
|---|---|---|---|
| P5 | P5-B1 online features; P5-D1 live webcam + phone stream; P5-D2 ONNX; P5-D3 A9; P5-D4 degraded mode; P5-A1 face tracking; P5-C1 only if latency fails; rehearsals | 42-50 | 18 |
| P6 | P6-C1 final runs + `make reproduce`; P6-C2 figures; error analyses; dataset v1.0 card; user study; threats to validity | 35 | 19 |
| P7 | Chapter drafts (I draft, you write and own the text), slides, demo video, cleanup | 24 | 50 |

**Totals:** about 435-445 C hours and about 200 M hours over 24 weeks, which means **about 8-9 h/week from you**. The original plan assumed roughly 4 × 8-10 h/week. Even with me writing the code, the human-only work (recording, annotation, guide meetings, report writing) does not shrink, so the scope guard should be agreed now, not in week 10. Suggested cut order if behind: B4/B5 reduced to one small config each, A8 dropped, dashboard kept as plain HTML, user study reduced to 3 people.

---

## 5. Human-only tasks

| ID | Task | Who | When (needed by) |
|---|---|---|---|
| H1 | Register for DMD and accept terms; download UAH-DriveSet; read both licences; give me local paths | Maruf | Start now; DMD approval may take days (week 1-2) |
| H2 | Install Python 3.11 (none on this machine) and confirm I may use `uv` for the lock file | Maruf | Before task 2 (week 1) |
| H3 | GPU route: Colab/Kaggle account; confirm the local RTX 3050 Laptop GPU (VRAM unknown, likely 4 GB) is usable for small runs | Maruf | Before P2 (week 7) |
| H4 | Pick the phone logging app; record a 5-minute parked test with tap-and-blink sync; send me the files | Maruf + helper | Week 2-3 |
| H5 | Guide meeting: approve plan, collection protocol, consent form, **AI-assisted development and its disclosure in the report**, and the solo arrangement given the four-member README | Maruf + Mrs. Patil | Week 2 |
| H6 | Recruit drivers (minimum 4) and a passenger for every trip. `06 §1` rule 4 requires a passenger to operate recording, so you cannot record alone | Maruf | Before pilot (week 4) |
| H7 | Signed consent forms; keep the name-to-ID mapping outside the repo | Maruf | Before each new participant |
| H8 | Recording trips (pilot, then 2 sessions/week) and trip sheets | Driver + passenger | Weeks 4-17 |
| H9 | Second annotator for the 20% overlap (kappa / interval IoU) | Another person | P2 (week 8-9) |
| H10 | Obtain PDFs of R9 (closest prior work), R10 (DMD), R6, R7 and Streiffer et al. (via college IEEE access) | Maruf | Week 2 |
| H11 | Decide whether the repo is public; if so, approve removing licensed PDFs (F2) | Maruf | Week 1 |
| H12 | Phone + car for live demo; demo rehearsals; user-study volunteers | Maruf + volunteers | P5-P6 |
| H13 | Tick "Verified by reading" in `01` only after you have read each paper yourself (`01` honesty rule) | Maruf | Before synopsis |

---

## 6. Top five risks for a solo build

| # | Risk (register ID) | Why it is worse solo | First action |
|---|---|---|---|
| 1 | **Paired data cannot be collected alone** (R1, R2, R9) | Safety rule 4 requires a passenger; the protocol needs 4 or more drivers and a second annotator. Without people there is no F1 setting, and A1/A4/A5 have no honest data | Recruit 2-3 helpers and book the guide meeting in week 1-2; build the own-data pipeline (#30) against your parked test log so the first real trip goes straight through |
| 2 | **Effort vs capacity** (R7, R15) | About 200 human hours over 24 weeks at about 8-9 h/week with no backup | Agree the cut list (§4) now; I keep `docs/PROGRESS.md` with actual vs estimated hours so slippage is visible at every gate |
| 3 | **Dataset access and surprises** (R17, verify items) | Registration delay blocks P1; UAH units/axes/50 km/h event gap already differ from the contract | Register DMD and download UAH this week; write the inspection scripts before the loaders; contract freeze waits for verification |
| 4 | **Evaluation definitions not fixed** (R3, R11, R19) | Undefined macro-F1 class sets, FA/h denominator, matching rule and LODO validation would make every later number disputable, and there is nobody to catch it in review | Accept F10, F15-F17 before P1; implement and unit-test the metrics library early (moved to #21) |
| 5 | **Toolchain on Windows** (R16) | No `make`, no Python 3.11, no torch, Mac support untested; CI is the only second machine | Windows shim for every Make target, CI matrix ubuntu + windows, `uv` lock file; MediaPipe/PyTorch versions pinned in ADR-0006 |

Two more that are not in the register and need your decision: **copyright** of the IEEE PDFs in a possibly public repo (G5), and **academic integrity**. A four-member B.Tech project built mostly by one student with AI help should be cleared with the guide and disclosed (H5).

---

## 7. Proposed fixes (accept or reject each)

I have not edited any existing doc. Contract and API changes would bump `contract_version` and need your approval.

**Repo and process**
- **F1** Re-layout: `Docs/` → `docs/` (two-step `git mv` because Windows ignores case), ADRs → `docs/decisions/`, `README (1).md` → `docs/decisions/README.md`, PR template → `.github/`, CONTRIBUTING → root.
- **F2** Remove the licensed IEEE PDFs from the repo (keep a reference list with DOIs; PDFs on the private drive). Rewrite history only if the repo is public.
- **F3** Write ADR-0006: Python 3.11, PyTorch (pinned), MediaPipe Tasks Face Landmarker, FastAPI + Pydantic v2, `uv` lock, ONNX Runtime, ruff, pytest.
- **F4** Create `docs/reports/README.md` (outline from the master prompt's report map) and `docs/reports/consent_form_template.md`.
- **F23** CONTRIBUTING solo mode: self-review checklist + CI-equivalent local checks replace the "different workstream" reviewer; squash-merge to `dev` by me; you review at gates.
- **F27** README licence line → "MIT (see LICENSE)", confirm with the guide (D8).

**Data contract (bump to 1.0 at freeze)**
- **F5** `src/driverisk/data/schema.py` is the code source of truth for names, counts, units, ranges and label IDs; a test checks it against `03`.
- **F6** Use `mask_v`/`mask_m` everywhere (drop the `m_v` symbol); module paths `src/driverisk/data/`.
- **F7** Vision features are per 0.1 s step. Clip embeddings travel as a separate tensor `e_clip [B, 5, Dc]`, each repeated over its 10 steps inside the model; `x_v` stays 16 features (amend ADR-0002).
- **F8** All-NaN allowed for a whole modality when `pairing` excludes it, and per dataset for `lane_offset`/`lateral_drift_5s` (declared in meta).
- **F9** No face → `face_visible=0`, `face_conf=0`, other 14 vision features NaN. PERCLOS computed over available history once at least 10 s exist, else NaN. `eye_closed` uses a global default EAR threshold from config until per-driver calibration; threshold and its source stored in meta.
- **F10** `y_cls`: majority over labelled steps; if fewer than 50% of steps are labelled or there is a tie → -1. `y_risk[h]`: 1 if a counted event's onset lies in (t_end, t_end + H]; -1 if the trip has no event annotation, if t_end + H exceeds the trip, or if the window overlaps an `uncertain` event. Events JSON gains `uncertain: bool` and `source`.
- **F11** Global driver key `<dataset>:<driver_id>` used for splits and personalization; API `drv_`/`trp_` IDs map to it through a server table; own raw files named `own_<driverid>_<yyyymmdd>_<nn>_*`.
- **F12** ADR-0007 taxonomy: classes {normal, aggressive, drowsy, distracted}; events {harsh_accel, harsh_brake, sharp_turn, lane_weave, micro_sleep, distraction_episode}. API alert `type` = event names for event alerts plus `drowsy`/`distracted`/`aggressive` for state alerts, with predictive alerts only for event types. Rename `aggressive_accel` → `harsh_accel`; add a `sharp_turn` template; `yawning` stays an evidence template.
- **F20** `manifest.csv` gains `quality` (`ok|low`); meta gains `posed_segments` and `parked_segments`, which are excluded from FA/h and from speed-context training.
- **F25** UAH loader rules written into `03`: G → m/s² (×9.80665), km/h → m/s, axis map (Z→x_long, Y→y_lat with sign checked on data), `yaw_rate` = d(unwrap(yaw))/dt, inertial events valid only above 50 km/h.

**Personalization, metrics, protocol**
- **F13** One `configs/personalization.yaml`: `n_ready = 600` windows (10 min) for status `ready`, prior strength `n0 = 20`, A4 calibration = first `K_cal = 2` normal trips; shrink variance, not std. UAH A4 is reported per driver with "not enough data" where only one normal evaluation trip remains.
- **F14** Absolute safety floor: global hard thresholds (for example micro-sleep ≥ 1 s, harsh braking above the config threshold) always alert, whatever the personal baseline.
- **F15** Macro-F1 over each dataset's own class subset, reported per dataset. FA/h: numerator = alerts matched to no event within [onset - H_max, offset + 2 s] (one-to-one, earliest alert); denominator = hours of windows with no event, no uncertain segment and not posed. On UAH, FA/h only over segments above 50 km/h.
- **F16** Nested LODO on UAH (1 test, 1 rotating validation, 4 training drivers). Normalization stats saved in each run folder, not in `configs/`.
- **F17** A1: "M2 mean exceeds the best single-modality mean by more than the larger of the two seed std". A3 uses one criterion (the `04 §3.4` one). Rename fusion settings to S1/S2/S3 to avoid F1-score confusion.
- **F24** Add a time-offset perturbation (±100, 250, 500 ms) to `04 §3.4`. "Phone moved" is simulated on raw IMU before feature extraction. Mask hysteresis (k consecutive steps) to prevent degraded/recovered flapping.

**Tasks and API**
- **F18** Move a minimal dashboard shell into P0 (P1-D3 extends it). The P0-C2 freeze sets `contract_version = "1.0"`. Add `A7.md` to P2-A3. Two targets: `make reproduce RUN=<id>` and `make reproduce-all`.
- **F19** WS auth via `?token=` query parameter; add WS message `recovered`; specify the missing request/response schemas, severity mapping and `confidence` meaning. Mock app at `src/driverisk/serving/mock_app.py` with a thin `scripts/mock_server.py`; snapshot at `tests/contract/openapi_snapshot.json` plus a JSON-schema test for WS messages.

**Literature notes**
- **F21** Correct `01`: R5 is temporal 2D evidence (not 3D-CNN) and supports ADR-0005; reword the R1 and R8 claims; fix "A3 vs A4/A5" → A2/A3 and drop cross-attention from R3's support; add Streiffer, Omerustaoglu and Ou (from R2) as related camera+sensor fusion work to read; name and dismiss Brain4Cars in ADR-0003; add Ultra-Fusion as R12 (preprint, SLAM) or remove it.
- **F22** Windows support: Makefile plus a `make.ps1` shim with the same targets (or document `mingw32-make`); CI runs on Windows too.
- **F26** Add the copyright and academic-integrity items from §6 to `09_risk_register.md` as R21 and R22.
