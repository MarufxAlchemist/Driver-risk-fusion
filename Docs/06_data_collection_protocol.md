# 06. Data Collection Protocol (Own Recordings)

Purpose: record a small but **paired** dataset (driver-face video + phone IMU + GPS, same trip, same clock) because no public dataset provides this (`03_data_contract.md` 3.4). Also record personalization data (several trips per driver).

Status: DRAFT v0.1. **Show to the guide and get approval before any recording.**

---

## 1. Safety rules (highest priority)

1. **Nobody tests unsafe driving on public roads.** No deliberate drowsiness, no real distraction (texting, phone calls, reaching) and no dangerous manoeuvres while driving in traffic.
2. **Posed behaviours are done only with the vehicle parked or in a safe, closed, low-traffic area**, with the engine in a safe state and a second person present.
3. **Aggressive driving data** means *moderately* firm acceleration and braking within traffic law and speed limits, on an empty or lightly used road, with a passenger present and the driver in full control. No speeding, no swerving.
4. **A passenger operates the recording** and watches the phone. The driver never touches the phone or the laptop while moving.
5. Stop immediately if the driver or passenger feels unsafe, tired or uncomfortable.
6. Only drivers with a valid licence drive. Passengers wear seatbelts. Follow all local traffic rules.
7. If any activity could be dangerous, ask the guide. When in doubt, do it in a parked or simulated setting.

Where real behaviour cannot be recorded safely (drowsiness, distraction, micro-sleep), rely on DMD annotations for real behaviour and use **posed** examples from our own recordings only as additional data. The report must say which parts are posed.

---

## 2. Consent and privacy

| Item | Rule |
|---|---|
| Participants | Adults (18+) who volunteer, starting with team members |
| Consent form | Written, covers face video, motion/GPS data, purpose (academic project), storage, right to withdraw and delete. Template in `docs/reports/consent_form_template.md` (to be created in P0) |
| Identity | Participants get a random ID (`d01`, `d02`, ...). The mapping from name to ID is kept by Gautam in a file **outside the repo** |
| Storage | Raw video on a team drive with access limited to the four members. Never in git. Encrypted disk where possible |
| What may be shared | Features, trained models, plots. Raw video or identifiable frames only in the final report with explicit permission |
| Retention | Delete raw video after the project is evaluated, unless participants agree to keep it |
| Withdrawal | A participant can ask to have their data removed. Keep the `trip_id` list per participant for this reason |
| GPS | Store trip route only with consent. Otherwise store speed and heading without coordinates |

---

## 3. Equipment

| Item | Choice | Note |
|---|---|---|
| Phone A (driver camera) | Phone with front camera, 1080p, 30 fps | Mounted on windshield or dashboard, camera facing the driver |
| IMU + GPS source | Same phone or a second phone fixed rigidly to the car | **Decision D6:** pick one logging app that exports timestamped CSV (candidates: Sensor Logger, phyphox or a custom app). Test on both Android and iOS before the first trip |
| Mount | Rigid suction or vent mount | Record mount position in the trip metadata. Do not move it mid-trip |
| Laptop | For monitoring only (passenger) | |
| Storage | 128 GB or more free | 1080p video at 30 fps is large; consider 720p |
| Power | Car charger | Keep the phone from overheating |
| Sync aid | None needed besides hands and the phone | See section 5 |

The phone running the camera and the IMU logger must have an up-to-date clock (network time). Record the app's timestamp format in the meta file.

---

## 4. What to record

### 4.1 Target volume

| Item | Minimum | Better |
|---|---|---|
| Drivers | 4 (team) | 8-10 |
| Trips per driver | 4 (at least 2 for calibration + 2 for test) | 6 |
| Trip length | 15-20 min | 30 min |
| Total | about 5 hours | about 20 hours |

### 4.2 Trip types

| Type | How | Labels produced |
|---|---|---|
| Normal driving | Routine drive on a familiar route, several times of day | `normal` |
| Moderately assertive driving | Passenger-approved, within the limit, empty road | `aggressive` events (harsh accel/brake/turn candidates), labeled by thresholds + manual check |
| Lane drift (simulated) | Only on a closed area or empty straight road with very small, safe drift within the lane | `lane_weave` candidates; skip if not safe |
| **Parked posed segments** | Driver seated, car parked: eyes-closed intervals (1-3 s), yawning, head nods, looking at the phone, looking at the side window, reaching toward the passenger seat | `micro_sleep`, `drowsy`, `distracted` posed labels |
| Degradation tests | Parked or driving normally: cover the camera briefly, tilt the phone, sunglasses, cap, low-light cabin | Camera degradation tests for the robustness evaluation |

Vary: time of day (morning, afternoon, evening), lighting, with and without glasses, route type (city, highway if safe).

### 4.3 Route planning

Choose 2-3 fixed routes near campus (low traffic, known roads), one of which has an empty straight stretch for acceleration/brake tests where legal and safe. Record route, time, weather and traffic in the trip log.

---

## 5. Session procedure (per trip)

**Before**
1. Passenger checks battery, storage and that clocks are in sync with network time.
2. Participant signs the consent form (first session only), confirms driver ID.
3. Mount the phone, check the driver's face is fully visible, and the horizon/camera angle matches the previous trips. Take a still photo of the mount for the log.
4. Start the IMU/GPS logger, then the camera recording.

**Sync event at the start (before moving)**
5. Driver blinks slowly three times and says "start", **then the passenger taps the phone firmly twice** (visible as a spike in the IMU, and audible or visible in video). Pause 2 seconds.

**During**
6. Drive the planned route. The passenger notes events (harsh braking, unusual events, interruptions) on the trip sheet with approximate time.

**Sync event at the end (after stopping)**
7. Repeat the tap-twice and blink pattern. This allows drift correction (`03_data_contract.md` section 8).

**After**
8. Stop recording. Name files `<driverid>_<yyyymmdd>_<nn>_{video,imu,gps}`. Back up to the team drive the same day.
9. Fill in the trip sheet (section 6) and a one-line notes field.

---

## 6. Trip sheet (CSV columns, one row per trip)

`trip_id, driver_id, date, start_time, route, road_type, weather, traffic, lighting, phone_model, app_version, mount_position, glasses, trip_type, planned_events, observed_events, sync_start_ok, sync_end_ok, notes, recorder`

---

## 7. Annotation

| Step | Who | Tool | Output |
|---|---|---|---|
| Auto labels from thresholds (harsh accel/brake/turn) | Workstream B | `scripts/auto_events.py` | Candidate events |
| Eye closure and posed segments | Workstream A | Simple annotation tool (for example a CSV of intervals, or an open-source annotator) | Interval list |
| Review | A second member | Spot check 20% of intervals | Agreement score |
| Final events | Gautam merges | | `<trip_id>_events.json` per contract |

Inter-annotator agreement (Cohen's kappa or interval IoU) is computed on the 20% overlap and reported.

Annotation guide:
- `micro_sleep`: eyes closed for 1.0-3.0 s with no blink recovery.
- `distraction_episode`: gaze off the forward zone for more than 2 s, or hand-device interaction visible.
- Events whose boundaries are unclear: mark `uncertain` and exclude from headline metrics.

---

## 8. Processing steps after recording

1. Convert IMU/GPS CSV and video to the common clock (`03_data_contract.md` 8).
2. Estimate offset and drift from the two sync events, store in meta.
3. Run `features/vision.py` and `features/motion.py`, then write parquet and manifest rows with `pairing = "paired"`.
4. Run the contract validator. Fix failures before any training.
5. Add the trip to the dataset card (`data/README.md`).

---

## 9. Quality checks (before a trip counts)

| Check | Pass condition |
|---|---|
| Face visible | `face_visible` at least 85% of steps in normal segments |
| IMU continuity | No gaps over 0.5 s |
| GPS | Fix present for at least 90% of the trip (or documented outage) |
| Sync | Start and end sync events both detected, residual offset < 100 ms |
| Labels | At least one event or an explicit "no events" in the trip sheet |

Failed trips are kept for the degradation experiments but flagged `quality = "low"`.

---

## 10. Timeline

| Week | Activity |
|---|---|
| 1-2 | Guide approval, consent template, choose logging app, test pipeline in a parked car |
| 3-4 | First pilot trips (2 drivers, 2 trips each), run the whole processing chain |
| 5-10 | Recording sessions, about 2 per week |
| 11-16 | Personalization trips (later trips per driver) |
| 17-19 | Fill gaps found by the first evaluations |
| 20 | Freeze dataset v1.0, write the dataset card |
